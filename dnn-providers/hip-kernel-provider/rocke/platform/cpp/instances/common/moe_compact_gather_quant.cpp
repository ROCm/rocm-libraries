// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99-style C-engine mirror of
 * rocke/instances/common/moe_compact_gather_quant.py.
 *
 * The build follows build_moe_compact_gather_quant top-to-bottom and preserves
 * Python's left-to-right IRBuilder call order. In particular, every inline
 * constant remains a fresh emitted operation.
 */
#include "rocke/instance_moe_compact_gather_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/arena.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h"

#define ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K 128
#define ROCKE_MOE_COMPACT_GATHER_QUANT_FP8_MAX 448.0
#define ROCKE_MOE_COMPACT_GATHER_QUANT_AMAX_FLOOR 1.0e-6

static void rocke_mcgq_set_reason(char* reason, size_t reason_cap, const char* message)
{
    rocke_spec_set_reason(reason, reason_cap, message);
}

/* Reproduce str(KeyError(...)) from ArchTarget.from_gfx for validation. */
static void rocke_mcgq_set_unknown_arch_reason(char* reason, size_t reason_cap, const char* arch)
{
    const char* const* arches;
    int count = 0;
    int index;
    int wrote;
    size_t position = 0;

    if(reason == NULL || reason_cap == 0)
    {
        return;
    }
    arches = rocke_known_arches(&count);
    wrote = snprintf(
        reason + position, reason_cap - position, "\"unknown gfx target '%s'; known: [", arch);
    if(wrote < 0)
    {
        reason[0] = '\0';
        return;
    }
    position += (size_t)wrote;
    if(position >= reason_cap)
    {
        reason[reason_cap - 1] = '\0';
        return;
    }
    for(index = 0; index < count; ++index)
    {
        wrote = snprintf(reason + position,
                         reason_cap - position,
                         "%s'%s'",
                         index == 0 ? "" : ", ",
                         arches[index]);
        if(wrote < 0)
        {
            reason[reason_cap - 1] = '\0';
            return;
        }
        position += (size_t)wrote;
        if(position >= reason_cap)
        {
            reason[reason_cap - 1] = '\0';
            return;
        }
    }
    snprintf(reason + position, reason_cap - position, "]. Add a row to arch_specs.json.\"");
}

rocke_moe_compact_gather_quant_spec_t rocke_moe_compact_gather_quant_spec_default(void)
{
    rocke_moe_compact_gather_quant_spec_t spec;
    spec.tokens = 0;
    spec.hidden = 0;
    spec.max_blocks = 0;
    spec.tile_m = 16;
    spec.input_dtype = "bf16";
    spec.block_size = 256;
    spec.vec = 4;
    spec.wave_size = 64;
    spec.name = "rocke_moe_compact_gather_quant";
    return spec;
}

int rocke_moe_compact_gather_quant_hidden_blocks(const rocke_moe_compact_gather_quant_spec_t* spec)
{
    return spec == NULL ? 0 : spec->hidden / ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K;
}

int rocke_moe_compact_gather_quant_qvecs_per_block(
    const rocke_moe_compact_gather_quant_spec_t* spec)
{
    if(spec == NULL || spec->vec == 0)
    {
        return 0;
    }
    return spec->tile_m * ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K / spec->vec;
}

int rocke_moe_compact_gather_quant_passes_per_thread(
    const rocke_moe_compact_gather_quant_spec_t* spec)
{
    if(spec == NULL || spec->block_size == 0)
    {
        return 0;
    }
    return rocke_moe_compact_gather_quant_qvecs_per_block(spec) / spec->block_size;
}

int rocke_moe_compact_gather_quant_output_rows(const rocke_moe_compact_gather_quant_spec_t* spec)
{
    return spec == NULL ? 0 : spec->max_blocks * spec->tile_m;
}

rocke_status_t rocke_moe_compact_gather_quant_kernel_name(
    const rocke_moe_compact_gather_quant_spec_t* spec, char* out, size_t out_cap)
{
    char tokens_part[32];
    char hidden_part[32];
    char max_blocks_part[32];
    char tile_m_part[32];
    char block_part[32];
    char vec_part[32];
    const char* parts[7];

    if(spec == NULL || out == NULL || spec->name == NULL || spec->input_dtype == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(tokens_part, sizeof(tokens_part), "T%d", spec->tokens);
    snprintf(hidden_part, sizeof(hidden_part), "H%d", spec->hidden);
    snprintf(max_blocks_part, sizeof(max_blocks_part), "MB%d", spec->max_blocks);
    snprintf(tile_m_part, sizeof(tile_m_part), "tm%d", spec->tile_m);
    snprintf(block_part, sizeof(block_part), "b%d", spec->block_size);
    snprintf(vec_part, sizeof(vec_part), "v%d", spec->vec);
    parts[0] = spec->input_dtype;
    parts[1] = tokens_part;
    parts[2] = hidden_part;
    parts[3] = max_blocks_part;
    parts[4] = tile_m_part;
    parts[5] = block_part;
    parts[6] = vec_part;
    return rocke_kernel_name_join(spec->name, parts, 7, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_moe_compact_gather_quant_is_valid_spec(const rocke_moe_compact_gather_quant_spec_t* spec,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    const rocke_arch_target_t* target;
    char message[ROCKE_ERR_MSG_CAP];
    long long qvecs_per_block;
    long bytes_lds;
    bool supported_block_size;

    if(spec == NULL)
    {
        rocke_mcgq_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MOE_COMPACT_GATHER_QUANT_DEFAULT_ARCH;
    }
    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_mcgq_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(spec->tokens <= 0 || spec->hidden <= 0 || spec->max_blocks <= 0)
    {
        rocke_mcgq_set_reason(
            reason, reason_cap, "tokens, hidden, and max_blocks must be positive");
        return false;
    }
    if(spec->hidden % ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K != 0)
    {
        snprintf(message,
                 sizeof(message),
                 "hidden (%d) must be divisible by %d",
                 spec->hidden,
                 ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    if(spec->tile_m <= 0)
    {
        snprintf(message, sizeof(message), "tile_m must be positive (got %d)", spec->tile_m);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    if(spec->input_dtype == NULL
       || (strcmp(spec->input_dtype, "f16") != 0 && strcmp(spec->input_dtype, "bf16") != 0))
    {
        if(spec->input_dtype == NULL)
        {
            snprintf(message, sizeof(message), "unsupported input_dtype None");
        }
        else
        {
            snprintf(message, sizeof(message), "unsupported input_dtype '%s'", spec->input_dtype);
        }
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    if(spec->vec != 4)
    {
        rocke_mcgq_set_reason(reason, reason_cap, "the packed FP8 conversion path requires vec=4");
        return false;
    }
    supported_block_size = spec->block_size == 64 || spec->block_size == 128
                           || spec->block_size == 256 || spec->block_size == 512
                           || spec->block_size == 1024;
    if(!supported_block_size)
    {
        snprintf(message, sizeof(message), "unsupported block_size %d", spec->block_size);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    if(spec->block_size > rocke_arch_max_threads_per_block(target))
    {
        snprintf(message,
                 sizeof(message),
                 "block_size %d > max_threads_per_block %d on %s",
                 spec->block_size,
                 rocke_arch_max_threads_per_block(target),
                 arch);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    qvecs_per_block = (long long)spec->tile_m * ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K / spec->vec;
    if(qvecs_per_block % spec->block_size != 0)
    {
        snprintf(message,
                 sizeof(message),
                 "tile_m*%d/vec (%lld) must be divisible by block_size (%d)",
                 ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K,
                 qvecs_per_block,
                 spec->block_size);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    bytes_lds = (long)spec->block_size * 4;
    if(!rocke_arch_fits_lds(target, bytes_lds))
    {
        snprintf(message,
                 sizeof(message),
                 "LDS budget %ld > %d cap on %s",
                 bytes_lds,
                 target->lds_capacity_bytes,
                 arch);
        rocke_mcgq_set_reason(reason, reason_cap, message);
        return false;
    }
    rocke_mcgq_set_reason(reason, reason_cap, "ok");
    return true;
}

static rocke_value_t*
    rocke_mcgq_fmax(rocke_ir_builder_t* b, rocke_value_t* lhs, rocke_value_t* rhs, void* user)
{
    (void)user;
    return rocke_b_fmax(b, lhs, rhs);
}

static void rocke_mcgq_readonly_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->noalias = true;
    opts->noalias_set = true;
    opts->readonly = true;
    opts->readonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

static void rocke_mcgq_writeonly_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->noalias = true;
    opts->noalias_set = true;
    opts->writeonly = true;
    opts->writeonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

rocke_kernel_def_t* rocke_build_moe_compact_gather_quant(
    rocke_ir_builder_t* b, const rocke_moe_compact_gather_quant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_arch_target_t* target;
        const rocke_type_t* input_ty;
        rocke_param_opts_t opts;
        rocke_value_t* source;
        rocke_value_t* sorted_token_ids;
        rocke_value_t* block_expert_ids;
        rocke_value_t* activation;
        rocke_value_t* activation_scale;
        rocke_value_t* tokens;
        rocke_value_t* tid;
        rocke_value_t* hidden_block;
        rocke_value_t* routed_block;
        rocke_value_t* block_expert;
        rocke_value_t* active_block;
        rocke_value_t* hidden_base;
        rocke_value_t* routed_row_base;
        rocke_value_t* lds;
        rocke_value_t* amax;
        rocke_value_t* block_amax;
        rocke_value_t* scale;
        rocke_value_t* inv_scale;
        rocke_value_t** cached;
        rocke_value_t** output_rows;
        rocke_value_t** local_columns;
        int lds_shape[1];
        int passes_per_thread;
        int pass_index;
        int index;
        char why[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = ROCKE_MOE_COMPACT_GATHER_QUANT_DEFAULT_ARCH;
        }
        if(!rocke_moe_compact_gather_quant_is_valid_spec(spec, arch, why, sizeof(why)))
        {
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "invalid moe compact-gather-quant spec for %s: %s", arch, why);
        }

        input_ty = rocke_b_io_ir_type(b, spec->input_dtype);
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

        rocke_mcgq_readonly_opts(&opts, 16);
        source = rocke_b_param(b, "X", rocke_ptr_type(b, input_ty, "global"), &opts);
        rocke_mcgq_readonly_opts(&opts, 4);
        sorted_token_ids
            = rocke_b_param(b, "SortedTokenIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
        block_expert_ids
            = rocke_b_param(b, "BlockExpertIds", rocke_ptr_type(b, rocke_i32(), "global"), &opts);
        rocke_mcgq_writeonly_opts(&opts, 16);
        activation = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_fp8e4m3(), "global"), &opts);
        rocke_mcgq_writeonly_opts(&opts, 4);
        activation_scale
            = rocke_b_param(b, "AScale", rocke_ptr_type(b, rocke_f32(), "global"), &opts);
        tokens = rocke_b_param(b, "tokens", rocke_i32(), NULL);
        (void)rocke_b_param(b, "hidden", rocke_i32(), NULL);

        tid = rocke_b_thread_id_x(b);
        hidden_block = rocke_b_block_id_x(b);
        routed_block = rocke_b_block_id_y(b);
        block_expert = rocke_b_global_load_i32(b, block_expert_ids, routed_block, 0);
        {
            rocke_value_t* zero = rocke_b_const_i32(b, 0);
            active_block = rocke_b_cmp_ge(b, block_expert, zero);
        }
        {
            rocke_value_t* group_k = rocke_b_const_i32(b, ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K);
            hidden_base = rocke_b_mul(b, hidden_block, group_k);
        }
        {
            rocke_value_t* tile_m = rocke_b_const_i32(b, spec->tile_m);
            routed_row_base = rocke_b_mul(b, routed_block, tile_m);
        }
        lds_shape[0] = spec->block_size;
        lds = rocke_b_smem_alloc(b, rocke_f32(), lds_shape, 1, "gather_amax");

        amax = rocke_b_const_f32(b, ROCKE_MOE_COMPACT_GATHER_QUANT_AMAX_FLOOR);
        passes_per_thread = rocke_moe_compact_gather_quant_passes_per_thread(spec);
        cached = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)passes_per_thread * (size_t)spec->vec * sizeof(*cached));
        output_rows = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)passes_per_thread * sizeof(*output_rows));
        local_columns = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, (size_t)passes_per_thread * sizeof(*local_columns));
        if(cached == NULL || output_rows == NULL || local_columns == NULL)
        {
            return (rocke_kernel_def_t*)rocke_i_set_err(
                b, ROCKE_ERR_OOM, "moe compact-gather-quant cache allocation failed");
        }

        for(pass_index = 0; pass_index < passes_per_thread; ++pass_index)
        {
            rocke_value_t* qvec;
            rocke_value_t* element;
            rocke_value_t* local_row;
            rocke_value_t* local_column;
            rocke_value_t* output_row;
            rocke_value_t* token;
            rocke_value_t* token_ge_zero;
            rocke_value_t* token_lt_tokens;
            rocke_value_t* token_in_range;
            rocke_value_t* valid_token;
            rocke_value_t* safe_token;
            rocke_value_t* source_row;
            rocke_value_t* source_column;
            rocke_value_t* source_offset;
            rocke_value_t* packed;
            rocke_value_t* values[4];
            rocke_value_t* absolute_values[4];
            rocke_value_t* vector_amax;

            {
                rocke_value_t* pass_offset = rocke_b_const_i32(b, pass_index * spec->block_size);
                qvec = rocke_b_add(b, tid, pass_offset);
            }
            {
                rocke_value_t* vec = rocke_b_const_i32(b, spec->vec);
                element = rocke_b_mul(b, qvec, vec);
            }
            {
                rocke_value_t* group_k
                    = rocke_b_const_i32(b, ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K);
                local_row = rocke_b_div(b, element, group_k);
            }
            {
                rocke_value_t* group_k
                    = rocke_b_const_i32(b, ROCKE_MOE_COMPACT_GATHER_QUANT_GROUP_K);
                local_column = rocke_b_mod(b, element, group_k);
            }
            output_row = rocke_b_add(b, routed_row_base, local_row);
            token = rocke_b_global_load_i32(b, sorted_token_ids, output_row, 0);
            {
                rocke_value_t* zero = rocke_b_const_i32(b, 0);
                token_ge_zero = rocke_b_cmp_ge(b, token, zero);
            }
            token_lt_tokens = rocke_b_cmp_lt(b, token, tokens);
            token_in_range = rocke_b_land(b, token_ge_zero, token_lt_tokens);
            valid_token = rocke_b_land(b, active_block, token_in_range);
            {
                rocke_value_t* zero = rocke_b_const_i32(b, 0);
                safe_token = rocke_b_select(b, valid_token, token, zero);
            }
            {
                rocke_value_t* hidden = rocke_b_const_i32(b, spec->hidden);
                source_row = rocke_b_mul(b, safe_token, hidden);
            }
            source_column = rocke_b_add(b, hidden_base, local_column);
            source_offset = rocke_b_add(b, source_row, source_column);
            packed = rocke_b_global_load_vN(b, source, source_offset, input_ty, spec->vec, 0);
            for(index = 0; index < spec->vec; ++index)
            {
                rocke_value_t* element_value = rocke_b_vec_extract(b, packed, index);
                rocke_value_t* promoted = rocke_b_cast_to_f32(b, element_value);
                rocke_value_t* zero = rocke_b_const_f32(b, 0.0);
                values[index] = rocke_b_select(b, valid_token, promoted, zero);
            }
            for(index = 0; index < spec->vec; ++index)
            {
                absolute_values[index] = rocke_b_fabs(b, values[index]);
            }
            vector_amax = rocke_tree_reduce(b, rocke_mcgq_fmax, NULL, absolute_values, spec->vec);
            amax = rocke_b_fmax(b, amax, vector_amax);
            for(index = 0; index < spec->vec; ++index)
            {
                cached[pass_index * spec->vec + index] = values[index];
            }
            output_rows[pass_index] = output_row;
            local_columns[pass_index] = local_column;
        }

        target = rocke_arch_target_from_gfx(arch);
        if(target->wave_size == spec->wave_size && spec->block_size % spec->wave_size == 0)
        {
            block_amax = rocke_block_lds_reduce_with_wave_prologue(
                b, amax, lds, tid, spec->block_size, ROCKE_REDUCE_MAX, spec->wave_size);
        }
        else
        {
            block_amax
                = rocke_block_lds_reduce(b, amax, lds, tid, spec->block_size, ROCKE_REDUCE_MAX);
        }
        {
            rocke_value_t* floor = rocke_b_const_f32(b, ROCKE_MOE_COMPACT_GATHER_QUANT_AMAX_FLOOR);
            rocke_value_t* safe_amax = rocke_b_fmax(b, block_amax, floor);
            rocke_value_t* fp8_rcp
                = rocke_b_const_f32(b, 1.0 / ROCKE_MOE_COMPACT_GATHER_QUANT_FP8_MAX);
            scale = rocke_b_fmul(b, safe_amax, fp8_rcp);
        }
        inv_scale = rocke_b_rcp_fast(b, scale);

        for(pass_index = 0; pass_index < passes_per_thread; ++pass_index)
        {
            rocke_value_t* scaled_values[4];
            rocke_value_t* scaled;
            rocke_value_t* quantized;
            rocke_value_t* output_row_base;
            rocke_value_t* output_column;
            rocke_value_t* output_offset;

            for(index = 0; index < spec->vec; ++index)
            {
                scaled_values[index]
                    = rocke_b_fmul(b, cached[pass_index * spec->vec + index], inv_scale);
            }
            scaled = rocke_b_vec_pack(b, scaled_values, spec->vec, rocke_f32());
            quantized = rocke_b_cvt_pk_fp8_f32x4(b, scaled);
            {
                rocke_value_t* hidden = rocke_b_const_i32(b, spec->hidden);
                output_row_base = rocke_b_mul(b, output_rows[pass_index], hidden);
            }
            output_column = rocke_b_add(b, hidden_base, local_columns[pass_index]);
            output_offset = rocke_b_add(b, output_row_base, output_column);
            rocke_b_global_store_vN(b, activation, output_offset, quantized, spec->vec, 0);
        }

        {
            rocke_value_t* tile_m = rocke_b_const_i32(b, spec->tile_m);
            rocke_value_t* writes_scale = rocke_b_cmp_lt(b, tid, tile_m);
            rocke_if_t gate = rocke_b_scf_if(b, writes_scale);
            rocke_b_region_enter(b, gate.then_region);
            {
                rocke_value_t* scale_row = rocke_b_add(b, routed_row_base, tid);
                rocke_value_t* hidden_blocks
                    = rocke_b_const_i32(b, rocke_moe_compact_gather_quant_hidden_blocks(spec));
                rocke_value_t* scale_row_base = rocke_b_mul(b, scale_row, hidden_blocks);
                rocke_value_t* scale_offset = rocke_b_add(b, scale_row_base, hidden_block);
                rocke_b_global_store(b, activation_scale, scale_offset, scale, 4);
            }
            rocke_b_region_leave(b);
        }

        rocke_b_ret(b);
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_moe_compact_gather_quant_new(
    rocke_ir_builder_t* b, const rocke_moe_compact_gather_quant_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_moe_compact_gather_quant_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_moe_compact_gather_quant(b, spec, arch);
    });
}

rocke_status_t
    rocke_moe_compact_gather_quant_grid(const rocke_moe_compact_gather_quant_spec_t* spec,
                                        int out[3])
{
    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = rocke_moe_compact_gather_quant_hidden_blocks(spec);
    out[1] = spec->max_blocks;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t
    rocke_moe_compact_gather_quant_signature(rocke_arena_t* arena,
                                             const rocke_moe_compact_gather_quant_spec_t* spec,
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
    rocke_signature_builder_ptr(&signature, "X", spec->input_dtype, NULL);
    rocke_signature_builder_ptr(&signature, "SortedTokenIds", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "BlockExpertIds", "i32", NULL);
    rocke_signature_builder_ptr(&signature, "A", "fp8e4m3", NULL);
    rocke_signature_builder_ptr(&signature, "AScale", "f32", NULL);
    rocke_signature_builder_scalar(&signature, "tokens", "i32");
    rocke_signature_builder_scalar(&signature, "hidden", "i32");
    return rocke_signature_builder_build(&signature, out_items, out_count);
}

rocke_status_t
    rocke_moe_compact_gather_quant_lower_to_llvm(const rocke_moe_compact_gather_quant_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    rocke_ir_builder_t builder;
    rocke_kernel_def_t* kernel;
    rocke_status_t status;

    memset(&builder, 0, sizeof(builder));
    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = ROCKE_MOE_COMPACT_GATHER_QUANT_DEFAULT_ARCH;
    }
    kernel = rocke_build_moe_compact_gather_quant_new(&builder, spec, arch);
    if(kernel == NULL)
    {
        const char* message = rocke_ir_builder_error(&builder);
        status = rocke_ir_builder_status(&builder);
        if(err != NULL && err_cap > 0)
        {
            snprintf(err,
                     err_cap,
                     "%s",
                     message != NULL && message[0] != '\0'
                         ? message
                         : "build_moe_compact_gather_quant failed");
        }
        rocke_ir_builder_free(&builder);
        return status == ROCKE_OK ? ROCKE_ERR_VALUE : status;
    }
    status = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&builder);
    return status;
}
