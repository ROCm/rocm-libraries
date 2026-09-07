// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99-style engine mirror of
 * rocke/instances/common/moe_topk_active_pack.py.
 *
 * The builder calls below are deliberately sequenced one statement at a time.
 * Python evaluates nested IRBuilder calls left-to-right, while C++ does not
 * prescribe function-argument evaluation order; explicit temporaries preserve
 * the Python op stream and SSA numbering.
 */
#include "rocke/instance_moe_topk_active_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.scan.h"

static const double ROCKE_MOE_TOPK_ACTIVE_PACK_NEG_INF = -3.4028234663852886e38;
static const double ROCKE_MOE_TOPK_ACTIVE_PACK_INDEX_SENTINEL = 16777216.0;
static const double ROCKE_MOE_TOPK_ACTIVE_PACK_LOG2_E = 1.4426950408889634;

[[noreturn]] static void rocke_i_moe_topk_active_pack_fail(rocke_status_t status,
                                                           const char* message)
{
    ckc::raise_status(status, message != NULL ? message : "");
}

static void
    rocke_i_moe_topk_active_pack_set_reason(char* reason, size_t reason_cap, const char* text)
{
    rocke_spec_set_reason(reason, reason_cap, text);
}

static rocke_value_t* rocke_i_moe_topk_active_pack_load_lds_i32(rocke_ir_builder_t* b,
                                                                rocke_value_t* ptr,
                                                                rocke_value_t* index)
{
    rocke_value_t* indices[1];
    rocke_value_t* loaded;
    indices[0] = index;
    loaded = rocke_b_smem_load_vN(b, ptr, indices, 1, rocke_i32(), 1);
    return rocke_b_vec_extract(b, loaded, 0);
}

static rocke_value_t* rocke_i_moe_topk_active_pack_load_lds_f32(rocke_ir_builder_t* b,
                                                                rocke_value_t* ptr,
                                                                rocke_value_t* index)
{
    rocke_value_t* indices[1];
    rocke_value_t* loaded;
    indices[0] = index;
    loaded = rocke_b_smem_load_vN(b, ptr, indices, 1, rocke_f32(), 1);
    return rocke_b_vec_extract(b, loaded, 0);
}

static void rocke_i_moe_topk_active_pack_smem_store(rocke_ir_builder_t* b,
                                                    rocke_value_t* ptr,
                                                    rocke_value_t* index,
                                                    rocke_value_t* value)
{
    rocke_value_t* indices[1];
    indices[0] = index;
    rocke_b_smem_store_vN(b, ptr, indices, 1, value, 1);
}

static rocke_value_t* rocke_i_moe_topk_active_pack_lds_atomic_add(rocke_ir_builder_t* b,
                                                                  rocke_value_t* ptr,
                                                                  rocke_value_t* index,
                                                                  rocke_value_t* value)
{
    rocke_value_t* indices[1];
    indices[0] = index;
    return rocke_b_lds_atomic_add(b, ptr, indices, 1, value, NULL);
}

rocke_moe_topk_active_pack_spec_t rocke_moe_topk_active_pack_spec_default(void)
{
    rocke_moe_topk_active_pack_spec_t spec;
    spec.tokens = 0;
    spec.experts = 0;
    spec.topk = 0;
    spec.tile_m = 16;
    spec.block_size = 1024;
    spec.num_expert_groups = 1;
    spec.topk_groups = 1;
    spec.renormalize = true;
    spec.name = "rocke_moe_topk_active_pack";
    return spec;
}

int rocke_moe_topk_active_pack_total_pairs(const rocke_moe_topk_active_pack_spec_t* spec)
{
    return spec != NULL ? spec->tokens * spec->topk : 0;
}

int rocke_moe_topk_active_pack_max_blocks(const rocke_moe_topk_active_pack_spec_t* spec)
{
    return rocke_moe_topk_active_pack_total_pairs(spec);
}

int rocke_moe_topk_active_pack_max_padded_pairs(const rocke_moe_topk_active_pack_spec_t* spec)
{
    if(spec == NULL)
    {
        return 0;
    }
    return rocke_moe_topk_active_pack_max_blocks(spec) * spec->tile_m;
}

int rocke_moe_topk_active_pack_max_blocks_per_expert(const rocke_moe_topk_active_pack_spec_t* spec)
{
    if(spec == NULL || spec->tile_m <= 0)
    {
        return 0;
    }
    return (spec->tokens + spec->tile_m - 1) / spec->tile_m;
}

rocke_status_t rocke_moe_topk_active_pack_kernel_name(const rocke_moe_topk_active_pack_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap)
{
    char tokens_part[32];
    char experts_part[32];
    char topk_part[32];
    char groups_part[32];
    char tile_part[32];
    char block_part[32];
    const char* parts[6];
    const char* flags[1] = {"rn"};
    int flag_on[1];

    if(spec == NULL || out == NULL || spec->name == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    snprintf(tokens_part, sizeof(tokens_part), "T%d", spec->tokens);
    snprintf(experts_part, sizeof(experts_part), "E%d", spec->experts);
    snprintf(topk_part, sizeof(topk_part), "K%d", spec->topk);
    snprintf(
        groups_part, sizeof(groups_part), "G%dx%d", spec->num_expert_groups, spec->topk_groups);
    snprintf(tile_part, sizeof(tile_part), "tm%d", spec->tile_m);
    snprintf(block_part, sizeof(block_part), "b%d", spec->block_size);
    parts[0] = tokens_part;
    parts[1] = experts_part;
    parts[2] = topk_part;
    parts[3] = groups_part;
    parts[4] = tile_part;
    parts[5] = block_part;
    flag_on[0] = spec->renormalize ? 1 : 0;
    return rocke_kernel_name_join(spec->name, parts, 6, flags, flag_on, 1, out, out_cap, NULL);
}

static bool rocke_i_moe_topk_active_pack_resolve_target(const char* arch,
                                                        const rocke_archtarget_t** out_target,
                                                        char* reason,
                                                        size_t reason_cap)
{
    const rocke_archtarget_t* target = rocke_archtarget_from_gfx(arch);
    if(target != NULL)
    {
        *out_target = target;
        return true;
    }

    {
        char known[512];
        const char* const* arches;
        int count = 0;
        int index;
        size_t pos = 0;

        arches = rocke_known_arches(&count);
        known[pos++] = '[';
        for(index = 0; index < count && arches != NULL && pos + 8 < sizeof(known); ++index)
        {
            int wrote = snprintf(
                known + pos, sizeof(known) - pos, "%s'%s'", index == 0 ? "" : ", ", arches[index]);
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
        if(reason != NULL && reason_cap > 0)
        {
            snprintf(reason,
                     reason_cap,
                     "\"unknown gfx target '%s'; known: %s. Add a row to arch_specs.json.\"",
                     arch,
                     known);
        }
    }
    return false;
}

bool rocke_moe_topk_active_pack_is_valid_spec(const rocke_moe_topk_active_pack_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap)
{
    const rocke_archtarget_t* target = NULL;
    char text[256];
    long bytes_lds;
    int total_pairs;

    if(spec == NULL)
    {
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, "spec is NULL");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(!rocke_i_moe_topk_active_pack_resolve_target(arch, &target, reason, reason_cap))
    {
        return false;
    }
    if(spec->tokens <= 0 || spec->experts <= 0 || spec->topk <= 0)
    {
        snprintf(text,
                 sizeof(text),
                 "tokens, experts, and topk must be positive (got %d, %d, %d)",
                 spec->tokens,
                 spec->experts,
                 spec->topk);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->topk > spec->experts)
    {
        snprintf(
            text, sizeof(text), "topk (%d) must be <= experts (%d)", spec->topk, spec->experts);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->topk > 32)
    {
        snprintf(text, sizeof(text), "topk (%d) must be <= 32", spec->topk);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->tile_m <= 0)
    {
        snprintf(text, sizeof(text), "tile_m must be positive (got %d)", spec->tile_m);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->block_size != 64 && spec->block_size != 128 && spec->block_size != 256
       && spec->block_size != 512 && spec->block_size != 1024)
    {
        snprintf(text, sizeof(text), "unsupported block_size %d", spec->block_size);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->block_size > rocke_archtarget_max_threads_per_block(target))
    {
        snprintf(text,
                 sizeof(text),
                 "block_size %d > max_threads_per_block %d on %s",
                 spec->block_size,
                 rocke_archtarget_max_threads_per_block(target),
                 arch);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->experts > spec->block_size)
    {
        snprintf(text,
                 sizeof(text),
                 "experts (%d) > block_size (%d); the single-workgroup scan requires one lane per "
                 "expert",
                 spec->experts,
                 spec->block_size);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    total_pairs = rocke_moe_topk_active_pack_total_pairs(spec);
    if(total_pairs > spec->block_size)
    {
        snprintf(text,
                 sizeof(text),
                 "tokens*topk (%d) > block_size (%d); the decode path supports one routed pair per "
                 "lane",
                 total_pairs,
                 spec->block_size);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    if(spec->num_expert_groups != 1 || spec->topk_groups != 1)
    {
        snprintf(text,
                 sizeof(text),
                 "v1 supports the one-group routing case only (got %d/%d)",
                 spec->num_expert_groups,
                 spec->topk_groups);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    bytes_lds = (long)spec->block_size * 4 + 3L * spec->experts * 4 + (long)total_pairs * (4 + 4);
    if(!rocke_archtarget_fits_lds(target, bytes_lds))
    {
        snprintf(text,
                 sizeof(text),
                 "LDS budget %ld > %d cap on %s",
                 bytes_lds,
                 target->lds_capacity_bytes,
                 arch);
        rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, text);
        return false;
    }
    rocke_i_moe_topk_active_pack_set_reason(reason, reason_cap, "ok");
    return true;
}

static void rocke_i_moe_topk_active_pack_readonly_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->noalias = true;
    opts->noalias_set = true;
    opts->readonly = true;
    opts->readonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

static void rocke_i_moe_topk_active_pack_writeonly_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->writeonly = true;
    opts->writeonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

rocke_kernel_def_t* rocke_build_moe_topk_active_pack(rocke_ir_builder_t* b,
                                                     const rocke_moe_topk_active_pack_spec_t* spec,
                                                     const char* arch)
{
    char reason[640];
    const rocke_type_t* f32_ptr;
    const rocke_type_t* i32_ptr;
    rocke_param_opts_t opts;
    rocke_value_t* logits;
    rocke_value_t* correction_bias;
    rocke_value_t* sorted_token_ids;
    rocke_value_t* sorted_topk_ids;
    rocke_value_t* sorted_weights;
    rocke_value_t* block_expert_ids;
    rocke_value_t* counts;
    rocke_value_t* block_offsets;
    rocke_value_t* num_blocks;
    rocke_value_t* routed_scale;
    rocke_value_t* tid;
    rocke_value_t* c_zero;
    rocke_value_t* c_one;
    rocke_value_t* c_experts;
    rocke_value_t* c_pairs;
    rocke_value_t* c_tile_m;
    rocke_value_t* c_neg_inf;
    rocke_value_t* c_index_sentinel;
    rocke_value_t* c_one_f32;
    rocke_value_t* c_neg_log2e;
    rocke_value_t* lds_reduce;
    rocke_value_t* lds_counts;
    rocke_value_t* lds_block_offsets;
    rocke_value_t* lds_counters;
    rocke_value_t* lds_ids;
    rocke_value_t* lds_weights;
    rocke_value_t* in_experts;
    rocke_value_t* in_blocks;
    rocke_value_t* lane_as_f32;
    rocke_value_t* in_pairs;
    int total_pairs;
    int max_padded_pairs;
    int max_blocks_per_expert;
    int init_passes;
    int init_pass;
    int topk_slot_host;
    int block;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(!rocke_moe_topk_active_pack_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_moe_topk_active_pack_fail(
            ROCKE_ERR_VALUE,
            ckc::format_error("invalid moe topk-active-pack spec for %s: %s", arch, reason)
                .c_str());
    }

    total_pairs = rocke_moe_topk_active_pack_total_pairs(spec);
    max_padded_pairs = rocke_moe_topk_active_pack_max_padded_pairs(spec);
    max_blocks_per_expert = rocke_moe_topk_active_pack_max_blocks_per_expert(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    f32_ptr = rocke_ptr_type(b, rocke_f32(), "global");
    i32_ptr = rocke_ptr_type(b, rocke_i32(), "global");

    rocke_i_moe_topk_active_pack_readonly_opts(&opts, 16);
    logits = rocke_b_param(b, "Logits", f32_ptr, &opts);
    correction_bias = rocke_b_param(b, "CorrectionBias", f32_ptr, &opts);

    rocke_i_moe_topk_active_pack_writeonly_opts(&opts, 4);
    sorted_token_ids = rocke_b_param(b, "SortedTokenIds", i32_ptr, &opts);
    sorted_topk_ids = rocke_b_param(b, "SortedTopkIds", i32_ptr, &opts);
    sorted_weights = rocke_b_param(b, "SortedWeights", f32_ptr, &opts);
    block_expert_ids = rocke_b_param(b, "BlockExpertIds", i32_ptr, &opts);
    counts = rocke_b_param(b, "Counts", i32_ptr, &opts);
    block_offsets = rocke_b_param(b, "BlockOffsets", i32_ptr, &opts);
    num_blocks = rocke_b_param(b, "NumBlocks", i32_ptr, &opts);
    (void)rocke_b_param(b, "tokens", rocke_i32(), NULL);
    (void)rocke_b_param(b, "experts", rocke_i32(), NULL);
    routed_scale = rocke_b_param(b, "routed_scale", rocke_f32(), NULL);

    tid = rocke_b_thread_id_x(b);
    c_zero = rocke_b_const_i32(b, 0);
    c_one = rocke_b_const_i32(b, 1);
    c_experts = rocke_b_const_i32(b, spec->experts);
    c_pairs = rocke_b_const_i32(b, total_pairs);
    c_tile_m = rocke_b_const_i32(b, spec->tile_m);
    c_neg_inf = rocke_b_const_f32(b, ROCKE_MOE_TOPK_ACTIVE_PACK_NEG_INF);
    c_index_sentinel = rocke_b_const_f32(b, ROCKE_MOE_TOPK_ACTIVE_PACK_INDEX_SENTINEL);
    c_one_f32 = rocke_b_const_f32(b, 1.0);
    c_neg_log2e = rocke_b_const_f32(b, -ROCKE_MOE_TOPK_ACTIVE_PACK_LOG2_E);

    {
        int shape[1];
        shape[0] = spec->block_size;
        lds_reduce = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, "topk_reduce");
        shape[0] = spec->experts;
        lds_counts = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "expert_counts");
        lds_block_offsets = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "expert_block_offsets");
        lds_counters = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "expert_counters");
        shape[0] = total_pairs;
        lds_ids = rocke_b_smem_alloc(b, rocke_i32(), shape, 1, "topk_ids");
        lds_weights = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, "topk_weights");
    }

    in_experts = rocke_b_cmp_lt(b, tid, c_experts);
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_experts);
        rocke_b_region_enter(b, gate.then_region);
        rocke_i_moe_topk_active_pack_smem_store(b, lds_counts, tid, c_zero);
        rocke_i_moe_topk_active_pack_smem_store(b, lds_counters, tid, c_zero);
        rocke_b_region_leave(b);
    }

    in_blocks = rocke_b_cmp_lt(b, tid, c_pairs);
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_blocks);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* sentinel = rocke_b_const_i32(b, -1);
            rocke_b_global_store(b, block_expert_ids, tid, sentinel, 4);
        }
        rocke_b_region_leave(b);
    }

    init_passes = (max_padded_pairs + spec->block_size - 1) / spec->block_size;
    for(init_pass = 0; init_pass < init_passes; ++init_pass)
    {
        rocke_value_t* c_pass = rocke_b_const_i32(b, init_pass * spec->block_size);
        rocke_value_t* output_index = rocke_b_add(b, tid, c_pass);
        rocke_value_t* c_output_size = rocke_b_const_i32(b, max_padded_pairs);
        rocke_value_t* in_output = rocke_b_cmp_lt(b, output_index, c_output_size);
        rocke_if_t gate = rocke_b_scf_if(b, in_output);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* token_sentinel = rocke_b_const_i32(b, -1);
            rocke_value_t* topk_sentinel;
            rocke_value_t* zero_weight;
            rocke_b_global_store(b, sorted_token_ids, output_index, token_sentinel, 4);
            topk_sentinel = rocke_b_const_i32(b, -1);
            rocke_b_global_store(b, sorted_topk_ids, output_index, topk_sentinel, 4);
            zero_weight = rocke_b_const_f32(b, 0.0);
            rocke_b_global_store(b, sorted_weights, output_index, zero_weight, 4);
        }
        rocke_b_region_leave(b);
    }

    {
        rocke_value_t* is_first = rocke_b_cmp_eq(b, tid, c_zero);
        rocke_if_t gate = rocke_b_scf_if(b, is_first);
        rocke_b_region_enter(b, gate.then_region);
        rocke_b_global_store(b, num_blocks, c_zero, c_zero, 4);
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);

    if(spec->block_size >= spec->tokens * 64)
    {
        rocke_value_t* lane = rocke_b_mod(b, tid, rocke_b_const_i32(b, 64));
        rocke_value_t* wave = rocke_b_div(b, tid, rocke_b_const_i32(b, 64));
        rocke_value_t* wave_valid = rocke_b_cmp_lt(b, wave, rocke_b_const_i32(b, spec->tokens));
        rocke_value_t* safe_token = rocke_b_select(b, wave_valid, wave, c_zero);
        rocke_value_t* candidate_ids[32];
        rocke_value_t* candidate_scores[32];
        rocke_value_t* candidate_weights[32];
        int candidates_per_lane = (spec->experts + 63) / 64;
        int candidate;

        for(candidate = 0; candidate < candidates_per_lane; ++candidate)
        {
            rocke_value_t* expert = rocke_b_add(b, lane, rocke_b_const_i32(b, candidate * 64));
            rocke_value_t* expert_in_range = rocke_b_cmp_lt(b, expert, c_experts);
            rocke_value_t* expert_valid = rocke_b_land(b, wave_valid, expert_in_range);
            rocke_value_t* safe_expert = rocke_b_select(b, expert_valid, expert, c_zero);
            rocke_value_t* token_expert_base = rocke_b_mul(b, safe_token, c_experts);
            rocke_value_t* logit_offset = rocke_b_add(b, token_expert_base, safe_expert);
            rocke_value_t* logit = rocke_b_global_load_f32(b, logits, logit_offset, 0);
            rocke_value_t* bias = rocke_b_global_load_f32(b, correction_bias, safe_expert, 0);
            rocke_value_t* exp_arg = rocke_b_fmul(b, c_neg_log2e, logit);
            rocke_value_t* exponential = rocke_b_exp2(b, exp_arg);
            rocke_value_t* denominator = rocke_b_fadd(b, c_one_f32, exponential);
            rocke_value_t* score = rocke_b_rcp(b, denominator);
            rocke_value_t* biased = rocke_b_fadd(b, score, bias);
            candidate_ids[candidate] = expert;
            candidate_weights[candidate] = score;
            candidate_scores[candidate] = rocke_b_select(b, expert_valid, biased, c_neg_inf);
        }

        for(topk_slot_host = 0; topk_slot_host < spec->topk; ++topk_slot_host)
        {
            rocke_value_t* best_score = c_neg_inf;
            rocke_value_t* best_id = rocke_b_const_i32(b, spec->experts);
            rocke_value_t* best_weight = rocke_b_const_f32(b, 0.0);
            for(candidate = 0; candidate < candidates_per_lane; ++candidate)
            {
                rocke_value_t* score = candidate_scores[candidate];
                rocke_value_t* expert = candidate_ids[candidate];
                rocke_value_t* greater = rocke_b_fcmp(b, "ogt", score, best_score);
                rocke_value_t* equal = rocke_b_fcmp(b, "oeq", score, best_score);
                rocke_value_t* lower_id = rocke_b_cmp_lt(b, expert, best_id);
                rocke_value_t* equal_and_lower = rocke_b_land(b, equal, lower_id);
                rocke_value_t* better = rocke_b_lor(b, greater, equal_and_lower);
                best_score = rocke_b_select(b, better, score, best_score);
                best_id = rocke_b_select(b, better, expert, best_id);
                best_weight = rocke_b_select(b, better, candidate_weights[candidate], best_weight);
            }
            {
                static const int xor_masks[] = {32, 16, 8, 4, 2, 1};
                int xi;
                for(xi = 0; xi < 6; ++xi)
                {
                    rocke_value_t* other_score
                        = rocke_b_warp_shuffle_xor(b, best_score, xor_masks[xi]);
                    rocke_value_t* other_id = rocke_b_warp_shuffle_xor(b, best_id, xor_masks[xi]);
                    rocke_value_t* other_weight
                        = rocke_b_warp_shuffle_xor(b, best_weight, xor_masks[xi]);
                    rocke_value_t* greater = rocke_b_fcmp(b, "ogt", other_score, best_score);
                    rocke_value_t* equal = rocke_b_fcmp(b, "oeq", other_score, best_score);
                    rocke_value_t* lower_id = rocke_b_cmp_lt(b, other_id, best_id);
                    rocke_value_t* equal_and_lower = rocke_b_land(b, equal, lower_id);
                    rocke_value_t* better = rocke_b_lor(b, greater, equal_and_lower);
                    best_score = rocke_b_select(b, better, other_score, best_score);
                    best_id = rocke_b_select(b, better, other_id, best_id);
                    best_weight = rocke_b_select(b, better, other_weight, best_weight);
                }
            }
            {
                rocke_value_t* wave_base = rocke_b_mul(b, wave, rocke_b_const_i32(b, spec->topk));
                rocke_value_t* pair
                    = rocke_b_add(b, wave_base, rocke_b_const_i32(b, topk_slot_host));
                rocke_value_t* lane_zero = rocke_b_cmp_eq(b, lane, c_zero);
                rocke_value_t* is_writer = rocke_b_land(b, wave_valid, lane_zero);
                rocke_if_t writer = rocke_b_scf_if(b, is_writer);
                rocke_b_region_enter(b, writer.then_region);
                rocke_i_moe_topk_active_pack_smem_store(b, lds_ids, pair, best_id);
                rocke_i_moe_topk_active_pack_smem_store(b, lds_weights, pair, best_weight);
                (void)rocke_i_moe_topk_active_pack_lds_atomic_add(b, lds_counts, best_id, c_one);
                rocke_b_region_leave(b);
            }
            for(candidate = 0; candidate < candidates_per_lane; ++candidate)
            {
                rocke_value_t* selected = rocke_b_cmp_eq(b, candidate_ids[candidate], best_id);
                candidate_scores[candidate]
                    = rocke_b_select(b, selected, c_neg_inf, candidate_scores[candidate]);
            }
        }
    }
    else
    {
        lane_as_f32 = rocke_b_sitofp_f32(b, tid);
        {
            rocke_value_t* token_upper = rocke_b_const_i32(b, spec->tokens);
            rocke_for_t token_loop = rocke_b_scf_for(b, c_zero, token_upper, c_one, "token");
            rocke_b_region_enter(b, token_loop.body);
            {
                rocke_value_t* token = token_loop.iv;
                rocke_value_t* expert_valid = rocke_b_cmp_lt(b, tid, c_experts);
                rocke_value_t* safe_expert = rocke_b_select(b, expert_valid, tid, c_zero);
                rocke_value_t* token_expert_base = rocke_b_mul(b, token, c_experts);
                rocke_value_t* logit_offset = rocke_b_add(b, token_expert_base, safe_expert);
                rocke_value_t* logit = rocke_b_global_load_f32(b, logits, logit_offset, 0);
                rocke_value_t* bias = rocke_b_global_load_f32(b, correction_bias, safe_expert, 0);
                rocke_value_t* neg_logit = rocke_b_fmul(b, c_neg_log2e, logit);
                rocke_value_t* exp = rocke_b_exp2(b, neg_logit);
                rocke_value_t* denominator = rocke_b_fadd(b, c_one_f32, exp);
                rocke_value_t* score = rocke_b_rcp(b, denominator);
                rocke_value_t* biased_score = rocke_b_fadd(b, score, bias);
                rocke_value_t* selected_score
                    = rocke_b_select(b, expert_valid, biased_score, c_neg_inf);
                rocke_value_t* topk_upper = rocke_b_const_i32(b, spec->topk);
                rocke_iter_arg_t iter_arg;
                rocke_for_t topk_loop;

                iter_arg.name = "selected_score";
                iter_arg.init = selected_score;
                topk_loop = rocke_b_scf_for_iter(
                    b, c_zero, topk_upper, c_one, &iter_arg, 1, "topk_slot", false, true);
                rocke_b_region_enter(b, topk_loop.body);
                {
                    rocke_value_t* topk_slot = topk_loop.iv;
                    rocke_value_t* current_score = topk_loop.iter_vars[0];
                    rocke_value_t* winning_score = rocke_block_lds_reduce(
                        b, current_score, lds_reduce, tid, spec->block_size, ROCKE_REDUCE_MAX);
                    rocke_value_t* score_matches
                        = rocke_b_fcmp(b, "oeq", current_score, winning_score);
                    rocke_value_t* is_max = rocke_b_land(b, expert_valid, score_matches);
                    rocke_value_t* candidate_index
                        = rocke_b_select(b, is_max, lane_as_f32, c_index_sentinel);
                    rocke_value_t* winning_index = rocke_block_lds_reduce(
                        b, candidate_index, lds_reduce, tid, spec->block_size, ROCKE_REDUCE_MIN);
                    rocke_value_t* index_matches
                        = rocke_b_fcmp(b, "oeq", lane_as_f32, winning_index);
                    rocke_value_t* is_winner = rocke_b_land(b, is_max, index_matches);
                    rocke_value_t* c_topk = rocke_b_const_i32(b, spec->topk);
                    rocke_value_t* token_pair_base = rocke_b_mul(b, token, c_topk);
                    rocke_value_t* pair = rocke_b_add(b, token_pair_base, topk_slot);

                    {
                        rocke_if_t winner_gate = rocke_b_scf_if(b, is_winner);
                        rocke_b_region_enter(b, winner_gate.then_region);
                        rocke_i_moe_topk_active_pack_smem_store(b, lds_ids, pair, tid);
                        rocke_i_moe_topk_active_pack_smem_store(b, lds_weights, pair, score);
                        (void)rocke_i_moe_topk_active_pack_lds_atomic_add(
                            b, lds_counts, tid, c_one);
                        rocke_b_region_leave(b);
                    }

                    {
                        rocke_value_t* next_score
                            = rocke_b_select(b, is_winner, c_neg_inf, current_score);
                        rocke_value_t* yielded[1];
                        yielded[0] = next_score;
                        rocke_b_scf_yield(b, yielded, 1);
                    }
                }
                rocke_b_region_leave(b);
            }
            rocke_b_region_leave(b);
        }
    }

    rocke_b_sync(b);

    in_pairs = rocke_b_cmp_lt(b, tid, c_pairs);
    {
        rocke_if_t gate = rocke_b_scf_if(b, in_pairs);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* c_topk_for_token = rocke_b_const_i32(b, spec->topk);
            rocke_value_t* token = rocke_b_div(b, tid, c_topk_for_token);
            rocke_value_t* c_topk_for_base = rocke_b_const_i32(b, spec->topk);
            rocke_value_t* token_base = rocke_b_mul(b, token, c_topk_for_base);
            rocke_value_t* weight = rocke_i_moe_topk_active_pack_load_lds_f32(b, lds_weights, tid);

            if(spec->renormalize)
            {
                rocke_value_t* weight_sum = rocke_b_const_f32(b, 0.0);
                for(topk_slot_host = 0; topk_slot_host < spec->topk; ++topk_slot_host)
                {
                    rocke_value_t* c_slot = rocke_b_const_i32(b, topk_slot_host);
                    rocke_value_t* slot_index = rocke_b_add(b, token_base, c_slot);
                    rocke_value_t* slot_weight
                        = rocke_i_moe_topk_active_pack_load_lds_f32(b, lds_weights, slot_index);
                    weight_sum = rocke_b_fadd(b, weight_sum, slot_weight);
                }
                {
                    rocke_value_t* inverse_sum = rocke_b_rcp(b, weight_sum);
                    weight = rocke_b_fmul(b, weight, inverse_sum);
                }
            }
            weight = rocke_b_fmul(b, weight, routed_scale);
            rocke_i_moe_topk_active_pack_smem_store(b, lds_weights, tid, weight);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);

    {
        rocke_if_t gate = rocke_b_scf_if(b, in_experts);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* count = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_counts, tid);
            rocke_value_t* c_round = rocke_b_const_i32(b, spec->tile_m - 1);
            rocke_value_t* rounded = rocke_b_add(b, count, c_round);
            rocke_value_t* blocks = rocke_b_div(b, rounded, c_tile_m);
            rocke_b_global_store(b, counts, tid, count, 4);
            rocke_i_moe_topk_active_pack_smem_store(b, lds_block_offsets, tid, blocks);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);

    rocke_block_exclusive_scan_i32(b, lds_block_offsets, tid, spec->block_size, spec->experts);

    {
        rocke_if_t gate = rocke_b_scf_if(b, in_experts);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* offset
                = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_block_offsets, tid);
            rocke_value_t* count = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_counts, tid);
            rocke_value_t* c_round = rocke_b_const_i32(b, spec->tile_m - 1);
            rocke_value_t* rounded = rocke_b_add(b, count, c_round);
            rocke_value_t* blocks = rocke_b_div(b, rounded, c_tile_m);
            rocke_b_global_store(b, block_offsets, tid, offset, 4);

            for(block = 0; block < max_blocks_per_expert; ++block)
            {
                rocke_value_t* c_block_for_cmp = rocke_b_const_i32(b, block);
                rocke_value_t* block_is_live = rocke_b_cmp_lt(b, c_block_for_cmp, blocks);
                rocke_if_t block_gate = rocke_b_scf_if(b, block_is_live);
                rocke_b_region_enter(b, block_gate.then_region);
                {
                    rocke_value_t* c_block_for_offset = rocke_b_const_i32(b, block);
                    rocke_value_t* block_index = rocke_b_add(b, offset, c_block_for_offset);
                    rocke_b_global_store(b, block_expert_ids, block_index, tid, 4);
                }
                rocke_b_region_leave(b);
            }
        }
        rocke_b_region_leave(b);
    }

    {
        rocke_value_t* c_last_expert = rocke_b_const_i32(b, spec->experts - 1);
        rocke_value_t* is_last_expert = rocke_b_cmp_eq(b, tid, c_last_expert);
        rocke_if_t gate = rocke_b_scf_if(b, is_last_expert);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* last_offset
                = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_block_offsets, tid);
            rocke_value_t* last_count
                = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_counts, tid);
            rocke_value_t* c_round = rocke_b_const_i32(b, spec->tile_m - 1);
            rocke_value_t* rounded = rocke_b_add(b, last_count, c_round);
            rocke_value_t* last_blocks = rocke_b_div(b, rounded, c_tile_m);
            rocke_value_t* total_blocks = rocke_b_add(b, last_offset, last_blocks);
            rocke_b_global_store(b, num_blocks, c_zero, total_blocks, 4);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);

    {
        rocke_if_t gate = rocke_b_scf_if(b, in_pairs);
        rocke_b_region_enter(b, gate.then_region);
        {
            rocke_value_t* expert = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_ids, tid);
            rocke_value_t* weight = rocke_i_moe_topk_active_pack_load_lds_f32(b, lds_weights, tid);
            rocke_value_t* local_offset
                = rocke_i_moe_topk_active_pack_lds_atomic_add(b, lds_counters, expert, c_one);
            rocke_value_t* expert_block
                = rocke_i_moe_topk_active_pack_load_lds_i32(b, lds_block_offsets, expert);
            rocke_value_t* expert_output_base = rocke_b_mul(b, expert_block, c_tile_m);
            rocke_value_t* output_index = rocke_b_add(b, expert_output_base, local_offset);
            rocke_value_t* c_topk_for_token = rocke_b_const_i32(b, spec->topk);
            rocke_value_t* token = rocke_b_div(b, tid, c_topk_for_token);
            rocke_value_t* c_topk_for_slot = rocke_b_const_i32(b, spec->topk);
            rocke_value_t* topk_slot = rocke_b_mod(b, tid, c_topk_for_slot);
            rocke_b_global_store(b, sorted_token_ids, output_index, token, 4);
            rocke_b_global_store(b, sorted_topk_ids, output_index, topk_slot, 4);
            rocke_b_global_store(b, sorted_weights, output_index, weight, 4);
        }
        rocke_b_region_leave(b);
    }

    rocke_b_ret(b);
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_moe_topk_active_pack_new(
    rocke_ir_builder_t* b, const rocke_moe_topk_active_pack_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_topk_active_pack_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_topk_active_pack(b, spec, arch);
    });
}

rocke_status_t rocke_moe_topk_active_pack_grid(const rocke_moe_topk_active_pack_spec_t* spec,
                                               int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = 1;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_moe_topk_active_pack_signature(rocke_arena_t* arena,
                                                    const rocke_moe_topk_active_pack_spec_t* spec,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count)
{
    rocke_signature_builder_t signature;
    rocke_status_t status;

    if(arena == NULL || spec == NULL || out_items == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    status = rocke_signature_builder_init(&signature, arena);
    if(status != ROCKE_OK)
    {
        return status;
    }
    rocke_signature_builder_ptr(&signature, "Logits", "f32", NULL);
    rocke_signature_builder_ptr(&signature, "CorrectionBias", "f32", NULL);
    rocke_signature_builder_ptr(&signature, "SortedTokenIds", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "SortedTopkIds", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "SortedWeights", "f32", NULL);
    rocke_signature_builder_ptr(&signature, "BlockExpertIds", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "Counts", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "BlockOffsets", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "NumBlocks", "i32", NULL);
    rocke_signature_builder_scalar(&signature, "tokens", "i32");
    rocke_signature_builder_scalar(&signature, "experts", "i32");
    rocke_signature_builder_scalar(&signature, "routed_scale", "f32");
    return rocke_signature_builder_build(&signature, out_items, out_count);
}

static void rocke_i_moe_topk_active_pack_copy_error(rocke_ir_builder_t* b,
                                                    char* err,
                                                    size_t err_cap,
                                                    const char* fallback)
{
    const char* message;
    if(err == NULL || err_cap == 0)
    {
        return;
    }
    message = rocke_ir_builder_error(b);
    snprintf(err, err_cap, "%s", message != NULL && message[0] != '\0' ? message : fallback);
}

rocke_status_t
    rocke_moe_topk_active_pack_lower_to_llvm(const rocke_moe_topk_active_pack_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t status;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    kernel = rocke_build_moe_topk_active_pack_new(&b, spec, arch);
    if(kernel == NULL)
    {
        status = rocke_ir_builder_status(&b);
        rocke_i_moe_topk_active_pack_copy_error(
            &b, err, err_cap, "build_moe_topk_active_pack failed");
        rocke_ir_builder_free(&b);
        return status == ROCKE_OK ? ROCKE_ERR_VALUE : status;
    }
    status = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return status;
}
