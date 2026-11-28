// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native authoring emitter for descriptor-driven global fragment transport.
 * Python twin: tests/helpers/test_fragment_io.py. */
#include "rocke/helper_rocke.helpers.tiling.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/dtypes.h"
#include "rocke/helper_rocke.helpers.mma_operand.h"
#include "rocke/instance_gfx950_block_scaled_gemm.h"
#include "rocke/instance_tf32_mma_probe.h"
#include "rocke/ir_serialize.h"
#include "rocke/ir_storage.h"
#include "rocke/mfma_scale_internal.h"
#include "rocke/wmma_scale_internal.h"

static bool parse_int(const char* text, int* out)
{
    char* end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if(errno || end == text || *end || value < INT_MIN || value > INT_MAX)
        return false;
    *out = int(value);
    return true;
}

static int print_kernel(const rocke_ir_builder_t* b)
{
    char* text = NULL;
    if(rocke_ir_serialize(b->kernel, &text) != ROCKE_OK)
        return 1;
    fputs(text, stdout);
    free(text);
    return 0;
}

static int tf32(int m, const char* mode)
{
    rocke_ir_builder_t b = {};
    auto* kernel = rocke_build_tf32_mma_probe(&b, m, mode);
    const int status = kernel && rocke_ir_builder_ok(&b) ? print_kernel(&b) : 1;
    if(status)
        fprintf(stderr, "%s\n", b.err);
    rocke_ir_builder_free(&b);
    return status;
}

static int gemm(const char* a, const char* bb, const char* c, const int* shape)
{
    auto spec = rocke_gfx950_block_scaled_gemm_spec_default();
    spec.name = "gemm_io";
    spec.M = shape[0];
    spec.N = shape[1];
    spec.K = shape[2];
    spec.dtype_a = a;
    spec.dtype_b = bb;
    spec.dtype_c = c;
    rocke_ir_builder_t b = {};
    auto* kernel = rocke_build_gfx950_block_scaled_gemm_new(&b, &spec, "gfx950");
    const int status = kernel && rocke_ir_builder_ok(&b) ? print_kernel(&b) : 1;
    if(status)
        fprintf(stderr, "%s\n", b.err);
    rocke_ir_builder_free(&b);
    return status;
}

static int prepare(const char* gfx,
                   const char* a,
                   const char* bb,
                   const char* role,
                   const char* conflicting_role = NULL)
{
    const auto* target = rocke_arch_target_from_gfx(gfx);
    if(!target || (strcmp(role, "a") != 0 && strcmp(role, "b") != 0))
        return 2;
    const bool mfma = strcmp(gfx, "gfx950") == 0;
    rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
    auto lookup = [&](const char* lhs, const char* rhs) {
        return rocke_mma_catalog_op_for_shape(&target->mma,
                                              mfma ? "mfma_scaled" : "wmma_scaled",
                                              lhs,
                                              rhs,
                                              "fp32",
                                              mfma ? 32 : 16,
                                              mfma ? 32 : 16,
                                              mfma ? 64 : 128,
                                              &scales);
    };
    const auto* atom = lookup(a, bb);
    if(!atom)
        return 2;
    auto supplied = *atom;
    if(conflicting_role)
    {
        const auto* fp8 = lookup("fp8e4m3", "fp8e4m3");
        if(!fp8)
            return 2;
        supplied.op_id = fp8->op_id;
    }
    const bool for_b = strcmp(role, "b") == 0;
    auto layout = mfma ? rocke_mfma_matrix_layout(for_b ? bb : a)
                       : rocke_scaled_wmma_matrix_layout(atom, for_b);
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "scaled_operand") != ROCKE_OK)
        return 2;
    rocke_tiling_tile_desc_t tile;
    rocke_tiling_fragment_t fragment;
    bool ok = rocke_h_make_matrix_fragment_desc(&b, &tile, &layout) == ROCKE_OK
              && rocke_h_make_fragment(
                     &b, &fragment, &tile, rocke_dtype_to_ir_type(for_b ? bb : a), NULL)
                     == ROCKE_OK
              && rocke_h_fill_fragment(&b, &fragment, 0) == ROCKE_OK;
    if(ok)
    {
        const auto before = b.kernel->body->num_ops;
        const auto* result = rocke_h_prepare_scaled_mma_operand(&b, &supplied, role, &fragment);
        if(conflicting_role)
        {
            char expected[112];
            snprintf(expected,
                     sizeof(expected),
                     "atom operand %s dtype='bf8e5m2' conflicts with catalog dtype='fp8e4m3'",
                     conflicting_role);
            ok = !result && rocke_ir_builder_status(&b) == ROCKE_ERR_VALUE
                 && strstr(b.err, expected) && b.kernel->body->num_ops == before;
        }
        else
            ok = result && rocke_ir_builder_ok(&b);
    }
    const int status = ok ? (conflicting_role ? 0 : print_kernel(&b)) : 1;
    if(status)
        fprintf(stderr, "unexpected %s %s preparation result: %s\n", gfx, role, b.err);
    rocke_ir_builder_free(&b);
    return status;
}

static int
    emit(const char* dtype, const int* config, bool print, bool batch = false, bool reverse = false)
{
    const int words = config[0], rows = config[1], groups = config[2];
    const int chunk = config[3], count = config[4], alignment = config[5];
    const int clip = config[6], dynamic = config[7], narrow = config[8];
    if(clip < 0 || clip > 1 || dynamic < 0 || dynamic > 2 || narrow < 0 || narrow > 1)
        return 2;
    const auto* info = rocke_dtype_info(dtype);
    const auto* logical = rocke_dtype_to_ir_type(dtype);
    if(!info || !logical || groups <= 0 || count <= 0 || rows <= 0 || rows > INT_MAX / 3
       || count > (INT_MAX / 3) / groups)
        return 2;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "fragment_io") != ROCKE_OK)
        return 2;
    auto* ptr
        = rocke_b_param(&b, "A", rocke_ptr_type(&b, rocke_storage_ir_type(dtype), "global"), NULL);
    const auto* output_dtype = narrow ? rocke_f16() : logical;
    auto* out = rocke_b_param(
        &b, "OUT", rocke_ptr_type(&b, rocke_storage_ir_type(output_dtype->name), "global"), NULL);
    rocke_bit_packing_t packing;
    rocke_fragment_packing_t fragment_packing;
    rocke_matrix_fragment_layout_t matrix;
    bool ok = rocke_bit_packing_init(&packing, info->encoded_bits, 0)
              && rocke_fragment_packing_init(&fragment_packing, &packing, count, 32, words)
              && rocke_matrix_fragment_layout_init(&matrix, &fragment_packing, chunk, groups, rows);
    rocke_tiling_tile_desc_t tile;
    ok = ok && rocke_h_make_matrix_fragment_desc(&b, &tile, &matrix) == ROCKE_OK;
    const int k = count * groups;
    const int lead = batch ? 1 : 0, rank = 2 + lead;
    int lengths[3] = {}, strides[3] = {};
    bool known[3] = {true, true, true};
    lengths[lead] = rows * 3;
    lengths[lead + 1] = k * 3;
    strides[lead] = k * 3;
    strides[lead + 1] = reverse ? -1 : 1;
    if(batch)
    {
        if(uint64_t(rows * 3) * (k * 3) > INT_MAX)
            return 2;
        known[0] = false;
        strides[0] = rows * 3 * k * 3;
    }
    rocke_tiling_tensor_desc_t tensor, destination;
    ok = ok
         && rocke_h_make_tensor_desc(&b, &tensor, lengths, strides, rank, logical, known)
                == ROCKE_OK;
    ok = ok
         && rocke_h_make_tensor_desc(&b, &destination, lengths, strides, rank, output_dtype, known)
                == ROCKE_OK;
    rocke_tiling_index_t origin[3] = {}, bounds[3] = {};
    bool bound_set[3] = {};
    origin[lead].constant = rows;
    origin[lead + 1].constant = k;
    bounds[lead].constant = rows * 2 - 1;
    bounds[lead + 1].constant = k * 2;
    bound_set[lead] = clip != 0;
    bound_set[lead + 1] = clip != 0 || dynamic == 2;
    if(batch)
        origin[0].value = rocke_b_param(&b, "batch", rocke_i32(), NULL);
    if(dynamic == 1)
        origin[lead + 1].value = rocke_b_param(&b, "origin_k", rocke_i32(), NULL);
    if(dynamic == 2)
        bounds[lead + 1].value = rocke_b_param(&b, "bound_k", rocke_i32(), NULL);
    auto* lane = rocke_b_thread_id_x(&b);
    rocke_tiling_window_t window, output_window;
    ok = ok
         && rocke_h_make_window(&b, &window, &tensor, origin, rank, bounds, bound_set) == ROCKE_OK;
    ok = ok
         && rocke_h_make_window(&b, &output_window, &destination, origin, rank, bounds, bound_set)
                == ROCKE_OK;
    rocke_tiling_fragment_t fragment;
    const bool packed_aligned = dynamic && info->encoded_bits < 8;
    ok = ok
         && rocke_h_load_fragment(
                &b, &fragment, ptr, &window, &tile, lane, alignment, packed_aligned)
                == ROCKE_OK;
    ok = ok
         && rocke_h_store_fragment(
                &b, out, &output_window, &fragment, lane, alignment, packed_aligned)
                == ROCKE_OK;
    if(!ok || !rocke_ir_builder_ok(&b))
    {
        fprintf(stderr, "%s\n", b.err);
        rocke_ir_builder_free(&b);
        return 1;
    }
    const int status = print ? print_kernel(&b) : 0;
    rocke_ir_builder_free(&b);
    return status;
}

static bool invalid_tile(bool overflow)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "invalid_tile") != ROCKE_OK)
        return false;
    const int row[] = {overflow ? 46341 : 2};
    const rocke_h_row_t axes[] = {{row, 1}, {row, 1}};
    const int major[] = {1, 2}, minor[] = {0, 0};
    const rocke_p_seq_t parts[] = {{major, minor, 2}};
    auto* layout
        = rocke_make_tile_distribution_encoding(&b, NULL, 0, axes, 2, parts, 1, NULL, NULL, 0);
    const int shape[] = {overflow ? row[0] : 1, row[0]};
    rocke_tiling_tile_desc_t tile;
    const bool ok = layout && rocke_h_make_tile_desc(&b, &tile, shape, 2, layout) == ROCKE_ERR_VALUE
                    && strstr(b.err, overflow ? "positive i32 range" : "distribution extent")
                    && b.kernel->body->num_ops == 0;
    if(!ok)
        fprintf(stderr, "unexpected tile validation result: %s\n", b.err);
    rocke_ir_builder_free(&b);
    return ok;
}

static bool scale_descriptor(const char* gfx, rocke_mma_scale_block_k_t block, bool mfma)
{
    const auto* target = rocke_arch_target_from_gfx(gfx);
    const rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", block};
    const int m = mfma ? 32 : 16, k = mfma ? 64 : 128;
    const auto* atom = rocke_mma_catalog_op_for_shape(
        &target->mma, mfma ? "mfma_scaled" : "wmma_scaled", "fp4", "fp8", "fp32", m, m, k, &scales);
    if(!atom)
        return false;
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "scale_descriptor") != ROCKE_OK)
        return false;
    rocke_tiling_tile_desc_t tile;
    const int groups = k / block;
    bool ok = rocke_h_make_mma_scale_desc(&b, &tile, atom, target) == ROCKE_OK && tile.rank == 2
              && tile.shape[0] == m && tile.shape[1] == groups
              && tile.register_count == (mfma ? 1 : groups) && b.kernel->body->num_ops == 0;
    auto stale = *atom;
    stale.srcs[1].scale_layout = NULL;
    ok = ok && rocke_h_make_mma_scale_desc(&b, &tile, &stale, target) == ROCKE_ERR_VALUE
         && strstr(b.err, "registered atom metadata");
    if(!ok)
        fprintf(stderr, "unexpected %s scale descriptor result: %s\n", gfx, b.err);
    rocke_ir_builder_free(&b);
    return ok;
}

static bool scale_refusal(const rocke_arch_target_t* target,
                          const rocke_mma_op_t* atom,
                          rocke_status_t status,
                          const char* expected)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "scale_refusal") != ROCKE_OK)
        return false;
    rocke_tiling_tile_desc_t tile;
    const bool ok = rocke_h_make_mma_scale_desc(&b, &tile, atom, target) == status
                    && strcmp(b.err, expected) == 0 && b.kernel->body->num_ops == 0;
    if(!ok)
        fprintf(stderr, "expected scale refusal '%s', got '%s'\n", expected, b.err);
    rocke_ir_builder_free(&b);
    return ok;
}

static bool scale_descriptor_refusals()
{
    const auto* gfx942 = rocke_arch_target_from_gfx("gfx942");
    const auto* tf32
        = rocke_mma_catalog_op_for_shape(&gfx942->mma, "mma", "tf32", "tf32", "fp32", 16, 16, 8);
    if(!tf32 || !scale_refusal(gfx942, NULL, ROCKE_ERR_TYPE, "scale descriptor requires an MmaOp")
       || !scale_refusal(gfx942, tf32, ROCKE_ERR_NOTIMPL, "unmodelled scale layout on 'gfx942'"))
        return false;

    auto target = *rocke_arch_target_from_gfx("gfx1250");
    const rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", ROCKE_MMA_SCALE_K32};
    const auto* atom = rocke_mma_catalog_op_for_shape(
        &target.mma, "wmma_scaled", "fp4", "fp8", "fp32", 16, 16, 128, &scales);
    const auto* profile = rocke_mma_catalog_op_for_shape(
        &target.mma, "wmma_scaled", "fp4", "fp4", "fp32", 16, 16, 128, &scales);
    if(!atom || !profile)
        return false;
    // Register our modified row so refusal happens in the profile check, not
    // the supplied-versus-registered metadata check.
    rocke_mma_op_t rows[] = {*atom, *profile};
    target.mma = {rows, 2};
    char message[256];
    rows[0].m = 32;
    snprintf(message, sizeof(message), "unmodelled scale shape for '%s'", atom->op_id);
    if(!scale_refusal(&target, &rows[0], ROCKE_ERR_NOTIMPL, message))
        return false;

    rows[0] = *atom;
    auto wrong_b = *atom->srcs[1].scale_layout;
    wrong_b.fn = atom->srcs[0].scale_layout->fn;
    rows[0].srcs[1].scale_layout = &wrong_b;
    snprintf(message, sizeof(message), "unmodelled scale_src1 map for '%s'", atom->op_id);
    if(!scale_refusal(&target, &rows[0], ROCKE_ERR_NOTIMPL, message))
        return false;

    rows[0] = *atom;
    target.mma.num_ops = 1;
    snprintf(message, sizeof(message), "unmodelled scale_src0 map for '%s'", atom->op_id);
    return scale_refusal(&target, &rows[0], ROCKE_ERR_NOTIMPL, message);
}

int main(int argc, char** argv)
{
    if(argc == 1)
    {
        const char* targets[] = {"gfx950", "gfx1250"};
        const char* roles[] = {"a", "b"};
        for(const char* gfx : targets)
            for(const char* role : roles)
                if(prepare(gfx, "bf8e5m2", "fp8e4m3", role, "a")
                   || prepare(gfx, "fp8e4m3", "bf8e5m2", role, "b"))
                    return 1;
        const int config[] = {8, 32, 2, 32, 32, 16, 0, 0, 0};
        return !invalid_tile(false) || !invalid_tile(true) || !scale_descriptor_refusals()
               || !scale_descriptor("gfx950", ROCKE_MMA_SCALE_K32, true)
               || !scale_descriptor("gfx1250", ROCKE_MMA_SCALE_K32, false)
               || !scale_descriptor("gfx1250", ROCKE_MMA_SCALE_K16, false)
               || emit("fp6e2m3", config, false);
    }
    if(argc == 4 && strcmp(argv[1], "--tf32") == 0)
    {
        int m;
        if(!parse_int(argv[2], &m))
            return 2;
        return tf32(m, argv[3]);
    }
    if(argc == 8 && strcmp(argv[1], "--gemm") == 0)
    {
        int shape[3];
        for(int i = 0; i < 3; ++i)
            if(!parse_int(argv[i + 5], &shape[i]))
                return 2;
        return gemm(argv[2], argv[3], argv[4], shape);
    }
    if(argc == 6 && strcmp(argv[1], "--prepare") == 0)
        return prepare(argv[2], argv[3], argv[4], argv[5]);
    if(argc != 12)
        return 2;
    const bool batch = strcmp(argv[1], "--emit-batch") == 0;
    const bool reverse = strcmp(argv[1], "--emit-reversed") == 0;
    if(!batch && !reverse && strcmp(argv[1], "--emit") != 0)
        return 2;
    int config[9];
    for(int i = 0; i < 9; ++i)
        if(!parse_int(argv[i + 3], &config[i]))
            return 2;
    return emit(argv[2], config, true, batch, reverse);
}
