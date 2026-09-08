// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_moe_sorting_instance_moe_sorting_spec_helpers_and_glue.c.c --
 * PUBLIC ENTRY / GLUE bucket of the chunked C99 port of
 * rocke/instances/common/moe_sorting.py.
 *
 * SCOPE (this TU only):
 *   spec surface          rocke_moe_sorting_spec_default / _total_pairs / _kernel_name
 *   validity gate         rocke_moe_sorting_is_valid_spec (public wrapper) +
 *                         rocke_moe_sort_is_valid_spec_impl (shared gate + wave_size)
 *   grid helpers          rocke_moe_sort_{histogram,scan,scatter,persistent}_grid
 *   signature builders    rocke_moe_sort_{histogram,scan,scatter,persistent}_signature
 *   workspace             rocke_moe_sorting_workspace_bytes
 *   shared module helpers rocke_moe_sort_decode_pair_token_topk,
 *                         rocke_moe_sort_decode_expert_load,
 *                         rocke_moe_sort_wave_kogge_stone_scan_i32
 *   public build entries  rocke_build_moe_sort_{histogram,scan,scatter,persistent}
 *                         (+ their _new init variants)
 *   lower convenience     rocke_build_moe_sort_*_lower_to_llvm
 *
 * The four build entries orchestrate: build ctx, populate inputs, then call the
 * matching prologue + phase functions (PEERS, declared in
 * rocke/instance_moe_sorting_internal.h, implemented in sibling TUs) in the exact
 * Python builder-call order; they return ctx->b->kernel.
 *
 * IR-free value/property helpers do not touch the builder; the three shared
 * module helpers + the build entries do, via the rocke_b_* surface, byte-faithful
 * to the Python op sequence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_from_gfx, .wave_size */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_kernel_name_join, sig entry, grid */
#include "rocke/helper_rocke.helpers.transforms.h" /* magic-division for pair decode */
#include "rocke/instance_moe_sorting.h"
#include "rocke/instance_moe_sorting_internal.h"
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 *  MoeSortingSpec value/property surface (IR-free).
 * ===================================================================== */

rocke_moe_sorting_spec_t rocke_moe_sorting_spec_default(void)
{
    rocke_moe_sorting_spec_t s;
    memset(&s, 0, sizeof(s));
    /* Required dims (tokens / topk / experts) have no Python default -> 0. */
    s.tokens = 0;
    s.topk = 0;
    s.experts = 0;
    /* @dataclass defaults. */
    s.block_size = 256;
    s.name = "rocke_moe_sorting";
    return s;
}

int rocke_moe_sorting_spec_total_pairs(const rocke_moe_sorting_spec_t* spec)
{
    /* @property total_pairs -> tokens * topk. */
    if(spec == NULL)
    {
        return 0;
    }
    return spec->tokens * spec->topk;
}

rocke_status_t rocke_moe_sorting_spec_kernel_name(const rocke_moe_sorting_spec_t* spec,
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
        return ROCKE_ERR_VALUE;
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

    return rocke_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

int rocke_moe_sorting_workspace_bytes(const rocke_moe_sorting_spec_t* spec)
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

bool rocke_moe_sort_is_valid_spec_impl(const rocke_moe_sorting_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap,
                                       int* out_wave_size)
{
    const rocke_archtarget_t* target;

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

    /* try: ArchTarget.from_gfx(arch) except KeyError as e: return False, str(e)
     *
     * str(KeyError(msg)) wraps the message in double-quotes, and _build_target
     * raises KeyError(
     *   f"unknown gfx target {gfx!r}; known: {sorted(specs)}. "
     *   f"Add a row to {_DATA_FILE.name}.")
     * -> reproduce the double-quoted repr with the sorted known-arch list. */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        char known[512];
        const char* const* arches;
        int count = 0;
        int k;
        size_t pos = 0;

        arches = rocke_known_arches(&count);
        known[pos++] = '[';
        for(k = 0; k < count && arches != NULL && pos + 8 < sizeof(known); ++k)
        {
            int wrote = snprintf(
                known + pos, sizeof(known) - pos, "%s'%s'", (k == 0) ? "" : ", ", arches[k]);
            if(wrote < 0)
            {
                break;
            }
            pos += (size_t)wrote;
        }
        if(pos + 1 < sizeof(known))
        {
            known[pos++] = ']';
        }
        known[pos] = '\0';
        CK_MOE_SORT_REJECT(
            "\"unknown gfx target '%s'; known: %s. Add a row to arch_specs.json.\"", arch, known);
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
    if(spec->block_size != 64 && spec->block_size != 128 && spec->block_size != 256
       && spec->block_size != 512 && spec->block_size != 1024)
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

bool rocke_moe_sorting_is_valid_spec(const rocke_moe_sorting_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    /* Public wrapper: same gate, wave_size discarded. */
    return rocke_moe_sort_is_valid_spec_impl(spec, arch, reason, reason_cap, NULL);
}

/* ===================================================================== *
 *  GRID HELPERS.
 *    histogram / scatter : ceil_div_grid((total_pairs, block_size)).
 *    scan / persistent   : (1, 1, 1).
 * ===================================================================== */

static rocke_status_t rocke_i_moe_sort_pairs_grid(const rocke_moe_sorting_spec_t* spec, int out[3])
{
    int totals[1];
    int tiles[1];

    if(spec == NULL || out == NULL || spec->block_size <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* ceil_div_grid((spec.total_pairs, spec.block_size)). */
    totals[0] = spec->tokens * spec->topk;
    tiles[0] = spec->block_size;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

rocke_status_t rocke_moe_sort_histogram_grid(const rocke_moe_sorting_spec_t* spec, int out[3])
{
    return rocke_i_moe_sort_pairs_grid(spec, out);
}

rocke_status_t rocke_moe_sort_scatter_grid(const rocke_moe_sorting_spec_t* spec, int out[3])
{
    return rocke_i_moe_sort_pairs_grid(spec, out);
}

rocke_status_t rocke_moe_sort_scan_grid(const rocke_moe_sorting_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* return (1, 1, 1) */
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_moe_sort_persistent_grid(const rocke_moe_sorting_spec_t* spec, int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* return (1, 1, 1) */
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

/* ===================================================================== *
 *  SIGNATURE (manifest) builders.
 *
 *  Each mirrors the Python SignatureBuilder().ptr(...).scalar(...).build()
 *  chain. rocke_sig_param / rocke_sig_scalar are exactly what the builder appends,
 *  so the emitted {name,type} sequence is byte-identical; the out[]/out_cap form
 *  matches this TU's public prototype.
 * ===================================================================== */

rocke_status_t rocke_moe_sort_histogram_signature(struct rocke_arena* arena,
                                                  const rocke_moe_sorting_spec_t* spec,
                                                  struct rocke_sig_entry* out,
                                                  size_t out_cap,
                                                  size_t* out_count)
{
    rocke_status_t st;

    (void)spec;
    if(arena == NULL || out == NULL || out_cap < 4)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_sig_param(arena, "TopkIds", "i32", NULL, &out[0]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "Hist", "i32", NULL, &out[1]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "num_pairs", "i32", &out[2]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "num_experts", "i32", &out[3]);
    if(st != ROCKE_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 4;
    }
    return ROCKE_OK;
}

rocke_status_t rocke_moe_sort_scan_signature(struct rocke_arena* arena,
                                             const rocke_moe_sorting_spec_t* spec,
                                             struct rocke_sig_entry* out,
                                             size_t out_cap,
                                             size_t* out_count)
{
    rocke_status_t st;

    (void)spec;
    if(arena == NULL || out == NULL || out_cap < 4)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_sig_param(arena, "Hist", "i32", NULL, &out[0]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "Offsets", "i32", NULL, &out[1]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "Counts", "i32", NULL, &out[2]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "num_experts", "i32", &out[3]);
    if(st != ROCKE_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 4;
    }
    return ROCKE_OK;
}

/* The 10-entry ABI shared by scatter + persistent (persistent is a superset). */
static rocke_status_t rocke_i_moe_sort_scatter_abi(struct rocke_arena* arena,
                                                   struct rocke_sig_entry* out,
                                                   size_t out_cap,
                                                   size_t* out_count)
{
    rocke_status_t st;

    if(arena == NULL || out == NULL || out_cap < 10)
    {
        return ROCKE_ERR_VALUE;
    }

    st = rocke_sig_param(arena, "TopkIds", "i32", NULL, &out[0]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "TopkWeights", "f32", NULL, &out[1]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "Offsets", "i32", NULL, &out[2]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "Counter", "i32", NULL, &out[3]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "SortedTokenIds", "i32", NULL, &out[4]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "SortedTopkIds", "i32", NULL, &out[5]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_param(arena, "SortedWeights", "f32", NULL, &out[6]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "tokens", "i32", &out[7]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "topk", "i32", &out[8]);
    if(st != ROCKE_OK)
    {
        return st;
    }
    st = rocke_sig_scalar(arena, "num_experts", "i32", &out[9]);
    if(st != ROCKE_OK)
    {
        return st;
    }

    if(out_count != NULL)
    {
        *out_count = 10;
    }
    return ROCKE_OK;
}

rocke_status_t rocke_moe_sort_scatter_signature(struct rocke_arena* arena,
                                                const rocke_moe_sorting_spec_t* spec,
                                                struct rocke_sig_entry* out,
                                                size_t out_cap,
                                                size_t* out_count)
{
    (void)spec;
    return rocke_i_moe_sort_scatter_abi(arena, out, out_cap, out_count);
}

rocke_status_t rocke_moe_sort_persistent_signature(struct rocke_arena* arena,
                                                   const rocke_moe_sorting_spec_t* spec,
                                                   struct rocke_sig_entry* out,
                                                   size_t out_cap,
                                                   size_t* out_count)
{
    /* Same 10 entries as scatter (superset ABI). */
    (void)spec;
    return rocke_i_moe_sort_scatter_abi(arena, out, out_cap, out_count);
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
void rocke_moe_sort_decode_pair_token_topk(rocke_ir_builder_t* b,
                                           rocke_value_t* pair_idx,
                                           int topk,
                                           rocke_value_t** out_t_idx,
                                           rocke_value_t** out_k_idx)
{
    rocke_value_t* tmp;
    rocke_value_t* rem;
    rocke_value_t* quot;

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
        rem = rocke_b_const_i32(b, 0);
        quot = tmp;
    }
    else
    {
        uint64_t mult = 0;
        int shift = 0;
        if(!rocke_calculate_magic_numbers(b, topk, &mult, &shift))
        {
            return; /* builder error already set */
        }
        quot = rocke_do_magic_division(b, tmp, mult, shift);
        rem = rocke_b_sub(b, tmp, rocke_b_mul(b, quot, rocke_b_const_i32(b, topk)));
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
void rocke_moe_sort_decode_expert_load(rocke_ir_builder_t* b,
                                       rocke_value_t* TopkIds,
                                       rocke_value_t* pair_idx,
                                       rocke_value_t* num_experts,
                                       rocke_value_t** out_eid,
                                       rocke_value_t** out_valid_e)
{
    rocke_value_t* eid;
    rocke_value_t* ge;
    rocke_value_t* lt;

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
    eid = rocke_b_global_load_i32(b, TopkIds, pair_idx, 0);
    ge = rocke_b_cmp_ge(b, eid, rocke_b_const_i32(b, 0));
    lt = rocke_b_cmp_lt(b, eid, num_experts);

    if(out_eid != NULL)
    {
        *out_eid = eid;
    }
    if(out_valid_e != NULL)
    {
        *out_valid_e = rocke_b_land(b, ge, lt);
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
rocke_value_t* rocke_moe_sort_wave_kogge_stone_scan_i32(rocke_ir_builder_t* b,
                                                        rocke_value_t* val,
                                                        int length,
                                                        rocke_value_t* lane_id)
{
    rocke_value_t* cur;
    int stride;

    if(b == NULL || val == NULL || lane_id == NULL)
    {
        return val;
    }

    cur = val;
    for(stride = 1; stride < length; stride *= 2)
    {
        rocke_value_t* c_stride = rocke_b_const_i32(b, stride);
        rocke_value_t* do_add = rocke_b_cmp_ge(b, lane_id, c_stride);
        /* Python evaluates select() args left-to-right: b.sub(...) emits its
         * SSA temp BEFORE b.const_i32(0). C leaves argument evaluation order
         * unspecified, so hoist the sub into its own statement to pin the
         * sub-then-const ordering and keep SSA value ids byte-identical
         * (otherwise the sub temp drifts +1, e.g. %sub11 -> %sub12). */
        rocke_value_t* src_sub = rocke_b_sub(b, lane_id, c_stride);
        rocke_value_t* src_lane = rocke_b_select(b, do_add, src_sub, rocke_b_const_i32(b, 0));
        rocke_value_t* addr = rocke_b_shl(b, src_lane, rocke_b_const_i32(b, 2));
        rocke_value_t* neighbour = rocke_b_ds_bpermute(b, addr, cur);
        /* Same left-to-right pin for the add temp inside the merge select. */
        rocke_value_t* merged = rocke_b_add(b, cur, neighbour);
        cur = rocke_b_select(b, do_add, merged, cur);
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

rocke_kernel_def_t* rocke_build_moe_sort_histogram(rocke_ir_builder_t* b,
                                                   const rocke_moe_sorting_spec_t* spec,
                                                   const char* arch)
{
    rocke_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950"; /* Python default: arch="gfx950" */
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: is_valid_spec gate, max_workgroup_size, BS/E, params, tid/bid,
     * pair_idx = bid*BS + tid. (lines 224-243) */
    if(!rocke_moe_sort_hist_prologue(&ctx))
    {
        return NULL;
    }

    /* stage 1: per-block LDS histogram. (lines 245-258) */
    rocke_moe_sort_hist_block_histogram(&ctx);

    /* stage 2 + return: merge LDS bins to global Hist. (lines 260-272) */
    return rocke_moe_sort_hist_merge_to_global(&ctx);
}

rocke_kernel_def_t* rocke_build_moe_sort_scan(rocke_ir_builder_t* b,
                                              const rocke_moe_sorting_spec_t* spec,
                                              const char* arch)
{
    rocke_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, resolve wave_size, max_workgroup_size, BS/E, params, tid,
     * c_E + in_bounds. (lines 363-384) */
    if(!rocke_moe_sort_scan_prologue(&ctx))
    {
        return NULL;
    }

    /* Path select on E <= wave_size (Python `if E <= wave_size:` line 386). */
    if(ctx.E <= ctx.wave_size)
    {
        return rocke_moe_sort_scan_wave_path(&ctx);
    }
    return rocke_moe_sort_scan_lds_path(&ctx);
}

rocke_kernel_def_t* rocke_build_moe_sort_scatter(rocke_ir_builder_t* b,
                                                 const rocke_moe_sorting_spec_t* spec,
                                                 const char* arch)
{
    rocke_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, max_workgroup_size, 10-entry ABI params, tid/bid,
     * pair_idx, _decode_pair_token_topk -> t_idx/k_idx, num_pairs, in_bounds.
     * (lines 481-525) */
    if(!rocke_moe_sort_scatter_prologue(&ctx))
    {
        return NULL;
    }

    /* scatter body + return. (lines 527-540) */
    return rocke_moe_sort_scatter_body(&ctx);
}

rocke_kernel_def_t* rocke_build_moe_sort_persistent(rocke_ir_builder_t* b,
                                                    const rocke_moe_sorting_spec_t* spec,
                                                    const char* arch)
{
    rocke_moe_sort_ctx_t ctx;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.b = b;
    ctx.spec = spec;
    ctx.arch = arch;

    /* prologue: gate, max_workgroup_size, BS/E/NP/n_pairs_per_thread, 10-entry
     * ABI params, tid, c_one/c_zero/c_E/c_BS/c_NP. (lines 639-682) */
    if(!rocke_moe_sort_persistent_prologue(&ctx))
    {
        return NULL;
    }

    /* phase 1: LDS histogram + write Counts. (lines 684-704) */
    rocke_moe_sort_persistent_histogram(&ctx);

    /* phase 2: in-place exclusive scan + write Offsets. (lines 706-713) */
    rocke_moe_sort_persistent_scan(&ctx);

    /* phase 3 + return: LDS scatter. (lines 715-741) */
    return rocke_moe_sort_persistent_scatter(&ctx);
}

/* ----- _new convenience init variants: init `b` with kernel_name(phase). ----- */

rocke_kernel_def_t* rocke_build_moe_sort_histogram_new(rocke_ir_builder_t* b,
                                                       const rocke_moe_sorting_spec_t* spec,
                                                       const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_sorting_spec_kernel_name(spec, "hist", name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_sort_histogram(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_moe_sort_scan_new(rocke_ir_builder_t* b,
                                                  const rocke_moe_sorting_spec_t* spec,
                                                  const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_sorting_spec_kernel_name(spec, "scan", name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_sort_scan(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_moe_sort_scatter_new(rocke_ir_builder_t* b,
                                                     const rocke_moe_sorting_spec_t* spec,
                                                     const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_sorting_spec_kernel_name(spec, "scatter", name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_sort_scatter(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_moe_sort_persistent_new(rocke_ir_builder_t* b,
                                                        const rocke_moe_sorting_spec_t* spec,
                                                        const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_sorting_spec_kernel_name(spec, "persistent", name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_sort_persistent(b, spec, arch);
    });
}

/* ===================================================================== *
 *  LOWER-TO-LLVM CONVENIENCE.
 *  build -> lower to LLVM .ll text. Each owns and frees its own IRBuilder.
 * ===================================================================== */

static void rocke_i_moe_sort_set_err(char* err, size_t err_cap, const char* msg)
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
static rocke_status_t
    rocke_i_moe_sort_lower(rocke_kernel_def_t* (*build_new)(rocke_ir_builder_t*,
                                                            const rocke_moe_sorting_spec_t*,
                                                            const char*),
                           const char* build_fail_msg,
                           const rocke_moe_sorting_spec_t* spec,
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
        rocke_i_moe_sort_set_err(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = build_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        rocke_i_moe_sort_set_err(err, err_cap, (m != NULL && m[0] != '\0') ? m : build_fail_msg);
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

rocke_status_t rocke_build_moe_sort_histogram_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                            const char* arch,
                                                            rocke_llvm_flavor_t flavor,
                                                            char** out_ll,
                                                            char* err,
                                                            size_t err_cap)
{
    return rocke_i_moe_sort_lower(rocke_build_moe_sort_histogram_new,
                                  "build_moe_sort_histogram failed",
                                  spec,
                                  arch,
                                  flavor,
                                  out_ll,
                                  err,
                                  err_cap);
}

rocke_status_t rocke_build_moe_sort_scan_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                       const char* arch,
                                                       rocke_llvm_flavor_t flavor,
                                                       char** out_ll,
                                                       char* err,
                                                       size_t err_cap)
{
    return rocke_i_moe_sort_lower(rocke_build_moe_sort_scan_new,
                                  "build_moe_sort_scan failed",
                                  spec,
                                  arch,
                                  flavor,
                                  out_ll,
                                  err,
                                  err_cap);
}

rocke_status_t rocke_build_moe_sort_scatter_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                          const char* arch,
                                                          rocke_llvm_flavor_t flavor,
                                                          char** out_ll,
                                                          char* err,
                                                          size_t err_cap)
{
    return rocke_i_moe_sort_lower(rocke_build_moe_sort_scatter_new,
                                  "build_moe_sort_scatter failed",
                                  spec,
                                  arch,
                                  flavor,
                                  out_ll,
                                  err,
                                  err_cap);
}

rocke_status_t rocke_build_moe_sort_persistent_lower_to_llvm(const rocke_moe_sorting_spec_t* spec,
                                                             const char* arch,
                                                             rocke_llvm_flavor_t flavor,
                                                             char** out_ll,
                                                             char* err,
                                                             size_t err_cap)
{
    return rocke_i_moe_sort_lower(rocke_build_moe_sort_persistent_new,
                                  "build_moe_sort_persistent failed",
                                  spec,
                                  arch,
                                  flavor,
                                  out_ll,
                                  err,
                                  err_cap);
}
