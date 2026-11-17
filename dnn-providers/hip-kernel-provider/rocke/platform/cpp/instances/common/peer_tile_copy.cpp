// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C ABI mirror of instances/common/peer_tile_copy.py.
 * Spec/default/validation/name -> rocke_peer_tile_copy_*.
 * build_peer_tile_copy -> rocke_build_peer_tile_copy_new, op-for-op.
 * Host-only Python problem validation is not a device IR operation.
 */
#include "rocke/instance_peer_tile_copy.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir_internal.h"

static const char* const peer_scalar_names[] = { "rows",
    "cols",
    "src_rows",
    "src_cols",
    "dst_rows",
    "dst_cols",
    "src_row",
    "src_col",
    "dst_row",
    "dst_col",
    "src_stride_row",
    "src_stride_col",
    "dst_stride_row",
    "dst_stride_col" };

rocke_peer_tile_copy_spec_t rocke_peer_tile_copy_spec_default(void)
{
    rocke_peer_tile_copy_spec_t spec;
    spec.tile_rows = 4;
    spec.tile_cols = 64;
    spec.block_size = 64;
    spec.dtype = "bf16";
    spec.name = "rocke_peer_tile_copy";
    return spec;
}

rocke_status_t rocke_peer_tile_copy_kernel_name(
    const rocke_peer_tile_copy_spec_t* spec, char* out, size_t out_cap)
{
    if(spec == NULL || spec->name == NULL || spec->dtype == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    int n = snprintf(out,
        out_cap,
        "%s_%s_%dx%d_b%d",
        spec->name,
        spec->dtype,
        spec->tile_rows,
        spec->tile_cols,
        spec->block_size);
    return n >= 0 && (size_t)n < out_cap ? ROCKE_OK : ROCKE_ERR_VALUE;
}

static bool peer_ascii_identifier(const char* name)
{
    if(name == NULL || !*name)
    {
        return false;
    }
    for(size_t i = 0; name[i]; ++i)
    {
        unsigned char c = (unsigned char)name[i];
        bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        if(!letter && !(i > 0 && c >= '0' && c <= '9'))
        {
            return false;
        }
    }
    return true;
}

bool rocke_peer_tile_copy_is_valid_spec(
    const rocke_peer_tile_copy_spec_t* spec, const char* arch, char* reason, size_t reason_cap)
{
    char message[256];
    const char* why = "ok";
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    if(spec == NULL)
    {
        why = "null spec";
    }
    else if(strcmp(arch, "gfx942") != 0 && strcmp(arch, "gfx950") != 0)
    {
        snprintf(message, sizeof(message), "unsupported peer tile copy arch '%s'", arch);
        why = message;
    }
    else if(spec->dtype == NULL || (strcmp(spec->dtype, "f16") && strcmp(spec->dtype, "bf16")))
    {
        if(spec->dtype == NULL)
        {
            why = "unsupported dtype None";
        }
        else
        {
            snprintf(message, sizeof(message), "unsupported dtype '%s'", spec->dtype);
            why = message;
        }
    }
    else if(spec->tile_rows <= 0 || spec->tile_cols <= 0)
    {
        why = "tile_rows and tile_cols must be positive integers";
    }
    else if(spec->block_size != 64 && spec->block_size != 128 && spec->block_size != 256)
    {
        why = "block_size must be in {64, 128, 256}";
    }
    else if((int64_t)spec->tile_rows * spec->tile_cols > 4096)
    {
        why = "tile area must not exceed 4096 elements";
    }
    else if(!peer_ascii_identifier(spec->name) || strlen(spec->name) > 255)
    {
        why = "name must be an ASCII identifier of at most 255 characters";
    }
    rocke_spec_set_reason(reason, reason_cap, why);
    return strcmp(why, "ok") == 0;
}

static rocke_kernel_def_t* peer_build(
    rocke_ir_builder_t* b, const rocke_peer_tile_copy_spec_t* spec, const char* arch)
{
    char reason[256];
    if(!rocke_peer_tile_copy_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid peer tile copy spec: %s", reason);
        return NULL;
    }
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);
    const rocke_type_t* dtype = rocke_b_io_ir_type(b, spec->dtype);
    rocke_param_opts_t options;
    memset(&options, 0, sizeof(options));
    options.noalias = true;
    options.noalias_set = true;
    options.readonly = true;
    options.readonly_set = true;
    options.align = 2;
    options.align_set = true;
    rocke_value_t* source = rocke_b_param(b, "X", rocke_ptr_type(b, dtype, "global"), &options);
    memset(&options, 0, sizeof(options));
    options.noalias = true;
    options.noalias_set = true;
    options.writeonly = true;
    options.writeonly_set = true;
    options.align = 2;
    options.align_set = true;
    rocke_value_t* destination
        = rocke_b_param(b, "Y", rocke_ptr_type(b, dtype, "global"), &options);
    rocke_value_t* args[14];
    for(int i = 0; i < 14; ++i)
    {
        args[i] = rocke_b_param(b, peer_scalar_names[i], rocke_i64(), NULL);
    }
    rocke_value_t* raw_tid = rocke_b_thread_id_x(b);
    rocke_value_t* tid = rocke_b_zext(b, raw_tid, rocke_i64());
    rocke_value_t* raw_col = rocke_b_block_id_x(b);
    rocke_value_t* block_col = rocke_b_zext(b, raw_col, rocke_i64());
    rocke_value_t* raw_row = rocke_b_block_id_y(b);
    rocke_value_t* block_row = rocke_b_zext(b, raw_row, rocke_i64());
    rocke_value_t* tile_rows = rocke_b_const_i64(b, spec->tile_rows);
    rocke_value_t* tile_cols = rocke_b_const_i64(b, spec->tile_cols);
    rocke_value_t* row_base = rocke_b_mul(b, block_row, tile_rows);
    rocke_value_t* col_base = rocke_b_mul(b, block_col, tile_cols);
    rocke_value_t* zero = rocke_b_const_i64(b, 0);
    int tile_area = spec->tile_rows * spec->tile_cols;
    rocke_value_t* area = rocke_b_const_i64(b, tile_area);
    for(int chunk = 0; chunk < (tile_area + spec->block_size - 1) / spec->block_size; ++chunk)
    {
        rocke_value_t* chunk_base = rocke_b_const_i64(b, chunk * spec->block_size);
        rocke_value_t* lane_element = rocke_b_add(b, tid, chunk_base);
        rocke_value_t* local_row = rocke_b_div(b, lane_element, tile_cols);
        rocke_value_t* row = rocke_b_add(b, row_base, local_row);
        rocke_value_t* local_col = rocke_b_mod(b, lane_element, tile_cols);
        rocke_value_t* col = rocke_b_add(b, col_base, local_col);
        rocke_value_t* src_row = rocke_b_add(b, args[6], row);
        rocke_value_t* src_col = rocke_b_add(b, args[7], col);
        rocke_value_t* dst_row = rocke_b_add(b, args[8], row);
        rocke_value_t* dst_col = rocke_b_add(b, args[9], col);
        rocke_value_t* valid = rocke_b_cmp_lt(b, lane_element, area);
        rocke_value_t* row_valid = rocke_b_cmp_lt(b, row, args[0]);
        valid = rocke_b_land(b, valid, row_valid);
        rocke_value_t* col_valid = rocke_b_cmp_lt(b, col, args[1]);
        valid = rocke_b_land(b, valid, col_valid);
        rocke_value_t* coordinates[] = { src_row, src_col, dst_row, dst_col };
        rocke_value_t* limits[] = { args[2], args[3], args[4], args[5] };
        for(int i = 0; i < 4; ++i)
        {
            rocke_value_t* nonnegative = rocke_b_cmp_ge(b, coordinates[i], zero);
            valid = rocke_b_land(b, valid, nonnegative);
            rocke_value_t* below_limit = rocke_b_cmp_lt(b, coordinates[i], limits[i]);
            valid = rocke_b_land(b, valid, below_limit);
        }
        rocke_if_t condition = rocke_b_scf_if(b, valid);
        rocke_b_region_enter(b, condition.then_region);
        rocke_value_t* src_row_offset = rocke_b_mul(b, src_row, args[10]);
        rocke_value_t* src_col_offset = rocke_b_mul(b, src_col, args[11]);
        rocke_value_t* src_offset = rocke_b_add(b, src_row_offset, src_col_offset);
        rocke_value_t* dst_row_offset = rocke_b_mul(b, dst_row, args[12]);
        rocke_value_t* dst_col_offset = rocke_b_mul(b, dst_col, args[13]);
        rocke_value_t* dst_offset = rocke_b_add(b, dst_row_offset, dst_col_offset);
        rocke_value_t* value = rocke_b_global_load(b, source, src_offset, dtype, 2);
        rocke_b_global_store(b, destination, dst_offset, value, 2);
        rocke_b_region_leave(b);
    }
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_peer_tile_copy_new(
    rocke_ir_builder_t* b, const rocke_peer_tile_copy_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b,
        [&]() -> rocke_kernel_def_t*
        {
            if(b == NULL)
            {
                return NULL;
            }
            char name[512];
            rocke_status_t status = rocke_peer_tile_copy_kernel_name(spec, name, sizeof(name));
            if(rocke_ir_builder_init(b, status == ROCKE_OK ? name : "rocke_peer_tile_copy")
                != ROCKE_OK)
            {
                return NULL;
            }
            return peer_build(b, spec, arch);
        });
}

rocke_status_t rocke_peer_tile_copy_grid(
    int64_t rows, int64_t cols, const rocke_peer_tile_copy_spec_t* spec, int out_grid[3])
{
    if(out_grid == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    out_grid[0] = 0;
    out_grid[1] = 0;
    out_grid[2] = 1;
    if(rows < 0 || cols < 0 || !rocke_peer_tile_copy_is_valid_spec(spec, NULL, NULL, 0))
    {
        return ROCKE_ERR_VALUE;
    }
    if(rows == 0 || cols == 0)
    {
        return ROCKE_OK;
    }
    int64_t x = cols / spec->tile_cols + (cols % spec->tile_cols != 0);
    int64_t y = rows / spec->tile_rows + (rows % spec->tile_rows != 0);
    if(x > INT_MAX || y > INT_MAX)
    {
        return ROCKE_ERR_VALUE;
    }
    out_grid[0] = (int)x;
    out_grid[1] = (int)y;
    return ROCKE_OK;
}

rocke_status_t rocke_peer_tile_copy_lower_to_llvm(const rocke_peer_tile_copy_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    if(out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    *out_ll = NULL;
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof(b));
    rocke_kernel_def_t* kernel = rocke_build_peer_tile_copy_new(&b, spec, arch);
    rocke_status_t status = kernel == NULL
        ? ROCKE_ERR_VALUE
        : rocke_lower_kernel_to_llvm(kernel, flavor, arch == NULL ? "gfx950" : arch, out_ll);
    if(kernel == NULL)
    {
        rocke_spec_set_reason(err, err_cap, b.err);
    }
    rocke_ir_builder_free(&b);
    return status;
}
