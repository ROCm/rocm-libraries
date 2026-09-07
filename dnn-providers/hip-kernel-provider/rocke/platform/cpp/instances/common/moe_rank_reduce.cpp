// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C engine mirror of rocke/instances/common/moe_rank_reduce.py.
 */
#include "rocke/instance_moe_rank_reduce.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.reduction.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h"

#define ROCKE_MOE_RANK_REDUCE_MAX_WORLD_SIZE 16

static void copy_reason(char* reason, size_t reason_cap, const char* text)
{
    if(reason != NULL && reason_cap > 0)
    {
        snprintf(reason, reason_cap, "%s", text != NULL ? text : "");
    }
}

rocke_moe_rank_reduce_rmsnorm_spec_t rocke_moe_rank_reduce_rmsnorm_spec_default(void)
{
    rocke_moe_rank_reduce_rmsnorm_spec_t spec;
    spec.width = 0;
    spec.world_size = 0;
    spec.dtype = "bf16";
    spec.block_size = 256;
    spec.vec = 2;
    spec.wave_size = 64;
    spec.fp32_internal = false;
    spec.name = "rocke_moe_rank_reduce_rmsnorm";
    return spec;
}

rocke_moe_rank_reduce_scatter_spec_t rocke_moe_rank_reduce_scatter_spec_default(void)
{
    rocke_moe_rank_reduce_scatter_spec_t spec;
    spec.width = 0;
    spec.world_size = 0;
    spec.dtype = "bf16";
    spec.block_size = 64;
    spec.vec = 2;
    spec.name = "rocke_moe_rank_reduce_scatter";
    return spec;
}

static rocke_status_t rank_reduce_kernel_name(const char* name,
                                              const char* dtype,
                                              int width,
                                              int world_size,
                                              int block_size,
                                              int vec,
                                              bool fp32_internal,
                                              bool include_flag,
                                              char* out,
                                              size_t out_cap)
{
    char n_part[32];
    char r_part[32];
    char b_part[32];
    char v_part[32];
    const char* parts[5];
    const char* flags[1] = {"f32"};
    int flag_on[1] = {fp32_internal ? 1 : 0};

    snprintf(n_part, sizeof(n_part), "N%d", width);
    snprintf(r_part, sizeof(r_part), "R%d", world_size);
    snprintf(b_part, sizeof(b_part), "b%d", block_size);
    snprintf(v_part, sizeof(v_part), "v%d", vec);
    parts[0] = dtype;
    parts[1] = n_part;
    parts[2] = r_part;
    parts[3] = b_part;
    parts[4] = v_part;
    return rocke_kernel_name_join(
        name, parts, 5, flags, flag_on, include_flag ? 1 : 0, out, out_cap, NULL);
}

rocke_status_t rocke_moe_rank_reduce_rmsnorm_kernel_name(
    const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    return rank_reduce_kernel_name(spec->name,
                                   spec->dtype,
                                   spec->width,
                                   spec->world_size,
                                   spec->block_size,
                                   spec->vec,
                                   spec->fp32_internal,
                                   true,
                                   out,
                                   out_cap);
}

rocke_status_t rocke_moe_rank_reduce_scatter_kernel_name(
    const rocke_moe_rank_reduce_scatter_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL)
        return ROCKE_ERR_VALUE;
    return rank_reduce_kernel_name(spec->name,
                                   spec->dtype,
                                   spec->width,
                                   spec->world_size,
                                   spec->block_size,
                                   spec->vec,
                                   false,
                                   false,
                                   out,
                                   out_cap);
}

static bool validate_common(int width,
                            int world_size,
                            const char* dtype,
                            int block_size,
                            int vec,
                            int n_per_block,
                            const char* arch,
                            char* reason,
                            size_t reason_cap,
                            bool require_even_partition)
{
    const rocke_archtarget_t* target;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int ok;

    if(arch == NULL)
        arch = "gfx950";
    if(reason != NULL && reason_cap > 0)
        reason[0] = '\0';
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        copy_reason(reason, reason_cap, arch);
        return false;
    }
    if(width <= 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason, reason_cap, "width must be > 0 (got %d)", width);
        return false;
    }
    if(world_size < 1 || world_size > ROCKE_MOE_RANK_REDUCE_MAX_WORLD_SIZE)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "world_size must be in [1, %d] (got %d)",
                     ROCKE_MOE_RANK_REDUCE_MAX_WORLD_SIZE,
                     world_size);
        return false;
    }
    if(require_even_partition && width % world_size != 0)
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "width (%d) must be divisible by world_size (%d)",
                     width,
                     world_size);
        return false;
    }
    if(rocke_arena_init(&arena, 4096) != ROCKE_OK)
        return false;
    rocke_io_spec_rule_init(&rule, dtype, block_size, vec);
    rule.n_per_block_set = 1;
    rule.n_per_block = n_per_block;
    rule.max_elems_per_thread_set = 1;
    rule.max_elems_per_thread = ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD;
    ok = rocke_validate_io(&arena, &rule, &why);
    if(!ok)
    {
        copy_reason(reason, reason_cap, why);
        rocke_arena_destroy(&arena);
        return false;
    }
    if(block_size > rocke_archtarget_max_threads_per_block(target))
    {
        if(reason != NULL && reason_cap > 0)
            snprintf(reason,
                     reason_cap,
                     "block_size %d > max_threads_per_block %d on %s",
                     block_size,
                     rocke_archtarget_max_threads_per_block(target),
                     arch);
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);
    return true;
}

bool rocke_is_valid_moe_rank_reduce_rmsnorm_spec(const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                                 const char* arch,
                                                 char* reason,
                                                 size_t reason_cap)
{
    const rocke_archtarget_t* target;
    long bytes_lds;
    if(spec == NULL)
        return false;
    if(!validate_common(spec->width,
                        spec->world_size,
                        spec->dtype,
                        spec->block_size,
                        spec->vec,
                        spec->width,
                        arch,
                        reason,
                        reason_cap,
                        false))
        return false;
    target = rocke_archtarget_from_gfx(arch != NULL ? arch : "gfx950");
    bytes_lds = (long)spec->block_size * 4;
    if(target == NULL || !rocke_archtarget_fits_lds(target, bytes_lds))
    {
        if(reason != NULL && reason_cap > 0 && target != NULL)
            snprintf(reason,
                     reason_cap,
                     "LDS budget %ld > %d cap on %s",
                     bytes_lds,
                     target->lds_capacity_bytes,
                     arch != NULL ? arch : "gfx950");
        return false;
    }
    copy_reason(reason, reason_cap, "ok");
    return true;
}

bool rocke_is_valid_moe_rank_reduce_scatter_spec(const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                                 const char* arch,
                                                 char* reason,
                                                 size_t reason_cap)
{
    if(spec == NULL)
        return false;
    if(spec->world_size <= 0)
    {
        copy_reason(reason, reason_cap, "world_size must be positive");
        return false;
    }
    if(!validate_common(spec->width,
                        spec->world_size,
                        spec->dtype,
                        spec->block_size,
                        spec->vec,
                        spec->width / spec->world_size,
                        arch,
                        reason,
                        reason_cap,
                        true))
        return false;
    copy_reason(reason, reason_cap, "ok");
    return true;
}

static rocke_value_t*
    rank_reduce_fadd(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user)
{
    (void)user;
    return rocke_b_fadd(b, a, c);
}

static void load_vec_as_f32(rocke_ir_builder_t* b,
                            rocke_value_t* ptr,
                            rocke_value_t* offset,
                            const rocke_type_t* dtype,
                            int vec,
                            rocke_value_t** out)
{
    rocke_value_t* packed = rocke_b_global_load_vN(b, ptr, offset, dtype, vec, 0);
    int i;
    for(i = 0; i < vec; ++i)
        out[i] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, packed, i));
}

static void store_vec_from_f32(rocke_ir_builder_t* b,
                               rocke_value_t* ptr,
                               rocke_value_t* offset,
                               rocke_value_t** values,
                               const rocke_type_t* dtype,
                               int vec)
{
    rocke_value_t* casted[8];
    rocke_value_t* packed;
    int i;
    for(i = 0; i < vec; ++i)
        casted[i] = rocke_b_cast_f32_to(b, values[i], dtype);
    packed = rocke_b_vec_pack(b, casted, vec, dtype);
    rocke_b_global_store_vN(b, ptr, offset, packed, vec, 0);
}

static void rank_reduce_vec(rocke_ir_builder_t* b,
                            rocke_value_t* partials,
                            rocke_value_t* row,
                            rocke_value_t* rows,
                            int width,
                            int world_size,
                            rocke_value_t* column,
                            const rocke_type_t* dtype,
                            int vec,
                            rocke_value_t** accum)
{
    rocke_value_t* c_width;
    int source_rank, i;
    for(i = 0; i < vec; ++i)
        accum[i] = rocke_b_const_f32(b, 0.0);
    c_width = rocke_b_const_i32(b, width);
    for(source_rank = 0; source_rank < world_size; ++source_rank)
    {
        rocke_value_t* rank_row;
        rocke_value_t* base;
        rocke_value_t* values[8];
        rank_row = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, source_rank), rows), row);
        base = rocke_b_add(b, rocke_b_mul(b, rank_row, c_width), column);
        load_vec_as_f32(b, partials, base, dtype, vec, values);
        for(i = 0; i < vec; ++i)
            accum[i] = rocke_b_fadd(b, accum[i], values[i]);
    }
}

static void add_readonly_ptr_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->noalias = true;
    opts->noalias_set = true;
    opts->readonly = true;
    opts->readonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

static void add_writeonly_ptr_opts(rocke_param_opts_t* opts, int align)
{
    memset(opts, 0, sizeof(*opts));
    opts->noalias = true;
    opts->noalias_set = true;
    opts->writeonly = true;
    opts->writeonly_set = true;
    opts->align = align;
    opts->align_set = true;
}

rocke_kernel_def_t* rocke_build_moe_rank_reduce_rmsnorm(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, const char* arch)
{
    const rocke_type_t* io_ty;
    const rocke_archtarget_t* target;
    rocke_param_opts_t opts;
    rocke_value_t *partials, *gamma, *output, *rows, *eps;
    rocke_value_t *tid, *row, *c_vec, *lds, *sum_sq, *total_sq, *inv_rms, *row_base;
    rocke_value_t* cached[ROCKE_REGISTER_TILE_MAX_ELEMS_PER_THREAD];
    int cached_count = 0;
    int chunks_per_thread;
    int chunk, i;
    char why[256];

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(!rocke_is_valid_moe_rank_reduce_rmsnorm_spec(spec, arch, why, sizeof(why)))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid moe rank-reduce RMSNorm spec for %s: %s", arch, why);
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    add_readonly_ptr_opts(&opts, 16);
    partials = rocke_b_param(b, "Partials", rocke_ptr_type(b, io_ty, "global"), &opts);
    gamma = rocke_b_param(b, "Gamma", rocke_ptr_type(b, io_ty, "global"), &opts);
    add_writeonly_ptr_opts(&opts, 16);
    output = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);
    rows = rocke_b_param(b, "rows", rocke_i32(), NULL);
    (void)rocke_b_param(b, "width", rocke_i32(), NULL);
    eps = rocke_b_param(b, "eps", rocke_f32(), NULL);

    tid = rocke_b_thread_id_x(b);
    row = rocke_b_block_id_x(b);
    c_vec = rocke_b_const_i32(b, spec->vec);
    {
        int shape[1] = {spec->block_size};
        lds = rocke_b_smem_alloc(b, rocke_f32(), shape, 1, "rank_reduce_rms");
    }

    chunks_per_thread = (spec->width / spec->block_size) / spec->vec;
    sum_sq = rocke_b_const_f32(b, 0.0);
    for(chunk = 0; chunk < chunks_per_thread; ++chunk)
    {
        rocke_value_t* column;
        rocke_value_t* reduced[8];
        rocke_value_t* squares[8];
        rocke_value_t* part;
        rocke_value_t* left = rocke_b_mul(b, rocke_b_const_i32(b, chunk * spec->block_size), c_vec);
        rocke_value_t* right = rocke_b_mul(b, tid, c_vec);
        column = rocke_b_add(b, left, right);
        rank_reduce_vec(b,
                        partials,
                        row,
                        rows,
                        spec->width,
                        spec->world_size,
                        column,
                        io_ty,
                        spec->vec,
                        reduced);
        if(!spec->fp32_internal)
        {
            for(i = 0; i < spec->vec; ++i)
                reduced[i] = rocke_b_cast_to_f32(b, rocke_b_cast_f32_to(b, reduced[i], io_ty));
        }
        for(i = 0; i < spec->vec; ++i)
            squares[i] = rocke_b_fmul(b, reduced[i], reduced[i]);
        part = rocke_tree_reduce(b, rank_reduce_fadd, NULL, squares, spec->vec);
        sum_sq = rocke_b_fadd(b, sum_sq, part);
        for(i = 0; i < spec->vec; ++i)
            cached[cached_count++] = reduced[i];
    }

    target = rocke_archtarget_from_gfx(arch);
    if(target->wave_size == spec->wave_size && spec->block_size % spec->wave_size == 0)
    {
        total_sq = rocke_block_lds_reduce_with_wave_prologue(
            b, sum_sq, lds, tid, spec->block_size, ROCKE_REDUCE_SUM, spec->wave_size);
    }
    else
    {
        total_sq = rocke_block_lds_reduce(b, sum_sq, lds, tid, spec->block_size, ROCKE_REDUCE_SUM);
    }
    inv_rms = rocke_b_rsqrt(
        b,
        rocke_b_fadd(
            b,
            rocke_b_fmul(b, total_sq, rocke_b_rcp(b, rocke_b_const_f32(b, (double)spec->width))),
            eps));
    row_base = rocke_b_mul(b, row, rocke_b_const_i32(b, spec->width));
    for(chunk = 0; chunk < chunks_per_thread; ++chunk)
    {
        rocke_value_t* column;
        rocke_value_t* gamma_values[8];
        rocke_value_t* normalized[8];
        rocke_value_t* left = rocke_b_mul(b, rocke_b_const_i32(b, chunk * spec->block_size), c_vec);
        rocke_value_t* right = rocke_b_mul(b, tid, c_vec);
        column = rocke_b_add(b, left, right);
        load_vec_as_f32(b, gamma, column, io_ty, spec->vec, gamma_values);
        for(i = 0; i < spec->vec; ++i)
        {
            rocke_value_t* scaled = rocke_b_fmul(b, inv_rms, gamma_values[i]);
            normalized[i] = rocke_b_fmul(b, cached[chunk * spec->vec + i], scaled);
        }
        store_vec_from_f32(
            b, output, rocke_b_add(b, row_base, column), normalized, io_ty, spec->vec);
    }
    rocke_b_ret(b);
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_moe_rank_reduce_scatter(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_scatter_spec_t* spec, const char* arch)
{
    const rocke_type_t* io_ty;
    rocke_param_opts_t opts;
    rocke_value_t *partials, *output, *rows, *rank;
    rocke_value_t *tid, *row, *c_vec, *shard_base, *output_row_base;
    int shard_width, chunks_per_thread, chunk;
    char why[256];

    if(b == NULL || spec == NULL)
        return NULL;
    if(arch == NULL)
        arch = "gfx950";
    if(!rocke_is_valid_moe_rank_reduce_scatter_spec(spec, arch, why, sizeof(why)))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "invalid moe rank-reduce scatter spec for %s: %s", arch, why);
        return NULL;
    }
    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

    add_readonly_ptr_opts(&opts, 16);
    partials = rocke_b_param(b, "Partials", rocke_ptr_type(b, io_ty, "global"), &opts);
    add_writeonly_ptr_opts(&opts, 16);
    output = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts);
    rows = rocke_b_param(b, "rows", rocke_i32(), NULL);
    (void)rocke_b_param(b, "width", rocke_i32(), NULL);
    rank = rocke_b_param(b, "rank", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);
    row = rocke_b_block_id_x(b);
    c_vec = rocke_b_const_i32(b, spec->vec);
    shard_width = spec->width / spec->world_size;
    shard_base = rocke_b_mul(b, rank, rocke_b_const_i32(b, shard_width));
    output_row_base = rocke_b_mul(b, row, rocke_b_const_i32(b, shard_width));
    chunks_per_thread = (shard_width / spec->block_size) / spec->vec;

    for(chunk = 0; chunk < chunks_per_thread; ++chunk)
    {
        rocke_value_t *local_column, *global_column;
        rocke_value_t* reduced[8];
        rocke_value_t* left = rocke_b_mul(b, rocke_b_const_i32(b, chunk * spec->block_size), c_vec);
        rocke_value_t* right = rocke_b_mul(b, tid, c_vec);
        local_column = rocke_b_add(b, left, right);
        global_column = rocke_b_add(b, shard_base, local_column);
        rank_reduce_vec(b,
                        partials,
                        row,
                        rows,
                        spec->width,
                        spec->world_size,
                        global_column,
                        io_ty,
                        spec->vec,
                        reduced);
        store_vec_from_f32(
            b, output, rocke_b_add(b, output_row_base, local_column), reduced, io_ty, spec->vec);
    }
    rocke_b_ret(b);
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_moe_rank_reduce_rmsnorm_new(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_moe_rank_reduce_rmsnorm_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_moe_rank_reduce_rmsnorm(b, spec, arch);
    });
}

rocke_kernel_def_t* rocke_build_moe_rank_reduce_scatter_new(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_scatter_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_moe_rank_reduce_scatter_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_moe_rank_reduce_scatter(b, spec, arch);
    });
}

static rocke_status_t row_grid(int rows, int out[3])
{
    int totals[2] = {rows, 1};
    int tiles[2] = {1, 1};
    if(out == NULL)
        return ROCKE_ERR_VALUE;
    return rocke_ceil_div_grid(totals, tiles, 2, out);
}

rocke_status_t rocke_moe_rank_reduce_rmsnorm_grid(int rows,
                                                  const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                                  int out[3])
{
    (void)spec;
    return row_grid(rows, out);
}

rocke_status_t rocke_moe_rank_reduce_scatter_grid(int rows,
                                                  const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                                  int out[3])
{
    (void)spec;
    return row_grid(rows, out);
}

rocke_status_t
    rocke_moe_rank_reduce_rmsnorm_signature(rocke_arena_t* arena,
                                            const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t status;
    if(arena == NULL || spec == NULL)
        return ROCKE_ERR_VALUE;
    status = rocke_signature_builder_init(&sb, arena);
    if(status != ROCKE_OK)
        return status;
    rocke_signature_builder_ptr(&sb, "Partials", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Gamma", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    rocke_signature_builder_scalar(&sb, "rows", "i32");
    rocke_signature_builder_scalar(&sb, "width", "i32");
    rocke_signature_builder_scalar(&sb, "eps", "f32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

rocke_status_t
    rocke_moe_rank_reduce_scatter_signature(rocke_arena_t* arena,
                                            const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count)
{
    rocke_signature_builder_t sb;
    rocke_status_t status;
    if(arena == NULL || spec == NULL)
        return ROCKE_ERR_VALUE;
    status = rocke_signature_builder_init(&sb, arena);
    if(status != ROCKE_OK)
        return status;
    rocke_signature_builder_ptr(&sb, "Partials", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    rocke_signature_builder_scalar(&sb, "rows", "i32");
    rocke_signature_builder_scalar(&sb, "width", "i32");
    rocke_signature_builder_scalar(&sb, "rank", "i32");
    return rocke_signature_builder_build(&sb, out_items, out_count);
}

static void copy_lower_error(rocke_ir_builder_t* b, char* err, size_t err_cap, const char* fallback)
{
    const char* message;
    if(err == NULL || err_cap == 0)
        return;
    message = rocke_ir_builder_error(b);
    snprintf(err, err_cap, "%s", message != NULL && message[0] != '\0' ? message : fallback);
}

rocke_status_t
    rocke_moe_rank_reduce_rmsnorm_lower_to_llvm(const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
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
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
        return ROCKE_ERR_VALUE;
    kernel = rocke_build_moe_rank_reduce_rmsnorm_new(&b, spec, arch);
    if(kernel == NULL)
    {
        status = rocke_ir_builder_status(&b);
        copy_lower_error(&b, err, err_cap, "build_moe_rank_reduce_rmsnorm failed");
        rocke_ir_builder_free(&b);
        return status == ROCKE_OK ? ROCKE_ERR_VALUE : status;
    }
    status = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch != NULL ? arch : "gfx950", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return status;
}

rocke_status_t
    rocke_moe_rank_reduce_scatter_lower_to_llvm(const rocke_moe_rank_reduce_scatter_spec_t* spec,
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
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
        return ROCKE_ERR_VALUE;
    kernel = rocke_build_moe_rank_reduce_scatter_new(&b, spec, arch);
    if(kernel == NULL)
    {
        status = rocke_ir_builder_status(&b);
        copy_lower_error(&b, err, err_cap, "build_moe_rank_reduce_scatter failed");
        rocke_ir_builder_free(&b);
        return status == ROCKE_OK ? ROCKE_ERR_VALUE : status;
    }
    status = rocke_lower_kernel_to_llvm_ex(
        kernel, flavor, arch != NULL ? arch : "gfx950", out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return status;
}
