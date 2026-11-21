// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors tiling/descriptors.py, fragments.py and layouts/matrix_fragment.py.
 * This bounded native surface supports static rank 1..8. Construction emits no IR.
 */
#include "rocke/helper_rocke.helpers.tiling.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/ir_storage.h"
#include "rocke/tiling_internal.h"

static void
    tuple_text(char* out, size_t capacity, const int* values, int count, const bool* known = NULL)
{
    size_t used = snprintf(out, capacity, "(");
    for(int i = 0; i < count && used < capacity; ++i)
        used += known && !known[i]
                    ? snprintf(out + used, capacity - used, "%sNone", i ? ", " : "")
                    : snprintf(out + used, capacity - used, "%s%d", i ? ", " : "", values[i]);
    if(used < capacity)
        snprintf(out + used, capacity - used, count == 1 ? ",)" : ")");
}

rocke_status_t rocke_h_make_tensor_desc(rocke_ir_builder_t* b,
                                        rocke_tiling_tensor_desc_t* out,
                                        const int* lengths,
                                        const int* strides,
                                        int rank,
                                        const rocke_type_t* dtype,
                                        const bool* has_length)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(rank < 1 || rank > ROCKE_TILING_MAX_RANK)
            ckc::raise_status(ROCKE_ERR_NOTIMPL, "native tensor descriptors support rank 1..8");
        if(!out || !lengths || !strides || !dtype)
            ckc::raise_status(ROCKE_ERR_VALUE, "null tensor descriptor argument");
        for(int i = 0; i < rank; ++i)
            if((!has_length || has_length[i]) && lengths[i] <= 0)
            {
                char text[256];
                tuple_text(text, sizeof(text), lengths, rank, has_length);
                throw ckc::ValueError(
                    ckc::format_error("lengths must be positive -- lengths=%s", text));
            }
        out->rank = rank;
        out->dtype = dtype;
        memcpy(out->lengths, lengths, rank * sizeof(int));
        for(int i = 0; i < rank; ++i)
            out->has_length[i] = !has_length || has_length[i];
        memcpy(out->strides, strides, rank * sizeof(int));
        return ROCKE_OK;
    });
}

rocke_status_t rocke_h_make_window(rocke_ir_builder_t* b,
                                   rocke_tiling_window_t* out,
                                   const rocke_tiling_tensor_desc_t* tensor,
                                   const rocke_tiling_index_t* origin,
                                   int origin_rank,
                                   const rocke_tiling_index_t* bounds,
                                   const bool* has_bound)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!out || !tensor || !origin)
            ckc::raise_status(ROCKE_ERR_VALUE, "null window argument");
        if(tensor->rank < 1 || tensor->rank > ROCKE_TILING_MAX_RANK)
            ckc::raise_status(ROCKE_ERR_NOTIMPL, "native tensor descriptors support rank 1..8");
        if(origin_rank != tensor->rank)
            throw ckc::ValueError(ckc::format_error(
                "window origin rank %d != tensor rank %d", origin_rank, tensor->rank));
        for(int axis = 0; axis < tensor->rank; ++axis)
            if(!origin[axis].value && origin[axis].constant < 0)
                throw ckc::ValueError(ckc::format_error(
                    "negative window origin on axis %d (origin=%d) -- a tile placed before "
                    "the buffer start generates out-of-bounds addresses; window origins must be "
                    "nonnegative",
                    axis,
                    origin[axis].constant));
        out->tensor = tensor;
        memcpy(out->origin, origin, tensor->rank * sizeof(*origin));
        for(int i = 0; i < tensor->rank; ++i)
        {
            out->has_bound[i] = bounds && (!has_bound || has_bound[i]);
            if(out->has_bound[i])
                out->bounds[i] = bounds[i];
        }
        return ROCKE_OK;
    });
}

rocke_status_t rocke_h_make_tile_desc(rocke_ir_builder_t* b,
                                      rocke_tiling_tile_desc_t* out,
                                      const int* shape,
                                      int rank,
                                      const rocke_tile_distribution_encoding_t* layout)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!out || !shape || !layout || rank < 1 || rank > ROCKE_TILING_MAX_RANK
           || layout->num_X != rank)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid tile descriptor");
        const int count = rocke_i_tiling_register_count(layout, shape);
        out->rank = rank;
        out->layout = layout;
        out->register_count = count;
        memcpy(out->shape, shape, rank * sizeof(int));
        return ROCKE_OK;
    });
}

rocke_status_t rocke_h_make_matrix_fragment_desc(rocke_ir_builder_t* b,
                                                 rocke_tiling_tile_desc_t* out,
                                                 const rocke_matrix_fragment_layout_t* layout)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        rocke_matrix_fragment_layout_t checked;
        if(!layout
           || !rocke_matrix_fragment_layout_init(&checked,
                                                 &layout->fragment,
                                                 layout->chunk_elements,
                                                 layout->lane_groups,
                                                 layout->lanes_per_group)
           || !layout->fragment.count
           || layout->fragment.count > uint64_t(INT_MAX) / layout->lane_groups)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid matrix fragment chunk layout");
        int free_levels[1], k_levels[3], free_count = 0, k_count = 0;
        int lane_major[2], lane_minor[2], lane_count = 0, reg_major[2], reg_minor[2], reg_count = 0;
        int free_lane = -1, group_lane = -1, repeat_level = -1, element_level = -1;
        if(layout->lanes_per_group > 1)
        {
            free_lane = free_count;
            free_levels[free_count++] = layout->lanes_per_group;
        }
        const int repeats = int(layout->fragment.count / layout->chunk_elements);
        if(repeats > 1)
        {
            repeat_level = k_count;
            k_levels[k_count++] = repeats;
        }
        if(layout->lane_groups > 1)
        {
            group_lane = k_count;
            k_levels[k_count++] = layout->lane_groups;
        }
        if(layout->chunk_elements > 1)
        {
            element_level = k_count;
            k_levels[k_count++] = layout->chunk_elements;
        }
        if(group_lane >= 0)
        {
            lane_major[lane_count] = 2;
            lane_minor[lane_count++] = group_lane;
        }
        if(free_lane >= 0)
        {
            lane_major[lane_count] = 1;
            lane_minor[lane_count++] = free_lane;
        }
        const int register_levels[] = {repeat_level, element_level};
        for(int i = 0; i < 2; ++i)
            if(register_levels[i] >= 0)
            {
                reg_major[reg_count] = 2;
                reg_minor[reg_count++] = register_levels[i];
            }
        rocke_h_row_t rows[] = {{free_levels, free_count}, {k_levels, k_count}};
        rocke_p_seq_t partitions[] = {{lane_major, lane_minor, lane_count}};
        auto* encoding = rocke_make_tile_distribution_encoding(
            b, NULL, 0, rows, 2, partitions, 1, reg_major, reg_minor, reg_count);
        if(!encoding)
            return ROCKE_ERR_VALUE;
        int shape[] = {layout->lanes_per_group, int(layout->fragment.count * layout->lane_groups)};
        return rocke_h_make_tile_desc(b, out, shape, 2, encoding);
    });
}

rocke_status_t rocke_h_make_mma_accumulator_desc(rocke_ir_builder_t* b,
                                                 rocke_tiling_tile_desc_t* out,
                                                 const rocke_mma_op_t* atom,
                                                 const rocke_arch_target_t* target,
                                                 bool result)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!atom)
            ckc::raise_status(ROCKE_ERR_TYPE, "accumulator descriptor requires an MmaOp");
        if(!target || !out)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid accumulator descriptor arguments");
        const auto* registered = rocke_mma_catalog_by_op_id(&target->mma, atom->op_id);
        if(!registered)
            throw ckc::ValueError(ckc::format_error("atom '%s' is not available on '%s'",
                                                    atom->op_id ? atom->op_id : "None",
                                                    target->gfx));
        const char* dtype = result ? atom->dst.dtype : atom->srcs[2].dtype;
        const int count = result ? atom->dst.frag_len : atom->srcs[2].frag_len;
        const auto* layout_map = result ? atom->dst.layout : atom->srcs[2].layout;
        const char* expected_dtype = result ? registered->dst.dtype : registered->srcs[2].dtype;
        const int expected_count = result ? registered->dst.frag_len : registered->srcs[2].frag_len;
        const auto* expected_map = result ? registered->dst.layout : registered->srcs[2].layout;
        const bool same_map = layout_map == expected_map
                              || (layout_map && expected_map && layout_map->fn == expected_map->fn
                                  && layout_map->frag_len == expected_map->frag_len
                                  && layout_map->wave_size == expected_map->wave_size);
        if(!atom->family || strcmp(atom->family, registered->family) != 0
           || atom->m != registered->m || atom->n != registered->n || atom->k != registered->k
           || atom->wave_size != registered->wave_size || !dtype
           || strcmp(dtype, expected_dtype) != 0 || count != expected_count || !same_map)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "accumulator descriptor requires the registered atom metadata");
        if(result)
            (void)rocke_mma_op_dst_layout(atom, b);
        else
            (void)rocke_mma_op_src_layout(atom, 2, b);
        if(!rocke_type_eq(rocke_dtype_to_ir_type(dtype), rocke_f32()))
            ckc::raise_status(ROCKE_ERR_NOTIMPL,
                              "accumulator descriptor currently supports fp32 metadata");
        const rocke_tile_distribution_encoding_t* encoding;
        if(strcmp(atom->family, "mma") == 0 || strcmp(atom->family, "mfma_scaled") == 0)
        {
            rocke_mfma_atom_t legacy = {atom->m,
                                        atom->n,
                                        atom->k,
                                        atom->srcs[0].frag_len,
                                        atom->srcs[1].frag_len,
                                        count,
                                        atom->srcs[0].dtype,
                                        "f32",
                                        atom->op_id};
            encoding = rocke_make_c_warp_dstr_encoding(b, &legacy);
        }
        else if((strcmp(atom->family, "wmma") == 0 || strcmp(atom->family, "wmma_scaled") == 0)
                && strcmp(target->gfx, "gfx1250") == 0)
        {
            if(atom->m != 16 || atom->n != 16 || atom->wave_size != 32 || count != 8)
                ckc::raise_status(ROCKE_ERR_NOTIMPL, "unmodelled gfx1250 accumulator layout");
            int m_levels[] = {atom->m / count, count}, n_levels[] = {atom->n};
            rocke_h_row_t rows[] = {{m_levels, 2}, {n_levels, 1}};
            int p_major[] = {1, 2}, p_minor[] = {0, 0};
            rocke_p_seq_t partitions[] = {{p_major, p_minor, 2}};
            int y_major[] = {1}, y_minor[] = {1};
            encoding = rocke_make_tile_distribution_encoding(
                b, NULL, 0, rows, 2, partitions, 1, y_major, y_minor, 1);
        }
        else
            throw ckc::NotImplError(
                ckc::format_error("unmodelled accumulator layout on '%s'", target->gfx));
        if(!encoding)
            return ROCKE_ERR_VALUE;
        int shape[] = {atom->m, atom->n};
        return rocke_h_make_tile_desc(b, out, shape, 2, encoding);
    });
}

rocke_status_t rocke_h_make_mma_scale_desc(rocke_ir_builder_t* b,
                                           rocke_tiling_tile_desc_t* out,
                                           const rocke_mma_op_t* atom,
                                           const rocke_arch_target_t* target)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!atom)
            ckc::raise_status(ROCKE_ERR_TYPE, "scale descriptor requires an MmaOp");
        if(!target || !out)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid scale descriptor arguments");
        const auto* registered = rocke_mma_catalog_by_op_id(&target->mma, atom->op_id);
        if(!registered || !atom->family || strcmp(atom->family, registered->family) != 0
           || atom->m != registered->m || atom->n != registered->n || atom->k != registered->k
           || atom->wave_size != registered->wave_size)
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "scale descriptor requires the registered atom metadata");
        for(int i = 0; i < 2; ++i)
        {
            const auto& src = atom->srcs[i];
            const auto& expected = registered->srcs[i];
            if(src.scale_block_size != expected.scale_block_size
               || src.scale_frag_len != expected.scale_frag_len
               || src.scale_layout != expected.scale_layout
               || bool(src.scale_dtype) != bool(expected.scale_dtype)
               || (src.scale_dtype && strcmp(src.scale_dtype, expected.scale_dtype) != 0))
                ckc::raise_status(ROCKE_ERR_VALUE,
                                  "scale descriptor requires the registered atom metadata");
        }
        const bool mfma
            = strcmp(target->gfx, "gfx950") == 0 && strcmp(atom->family, "mfma_scaled") == 0;
        const bool wmma
            = strcmp(target->gfx, "gfx1250") == 0 && strcmp(atom->family, "wmma_scaled") == 0;
        if(!mfma && !wmma)
            throw ckc::NotImplError(
                ckc::format_error("unmodelled scale layout on '%s'", target->gfx));
        const int block = atom->srcs[0].scale_block_size;
        const int m = mfma ? 32 : 16, k = mfma ? 64 : 128, wave = mfma ? 64 : 32;
        if(atom->m != m || atom->n != m || atom->k != k || atom->wave_size != wave
           || (block != 32 && !(wmma && block == 16)))
            throw ckc::NotImplError(
                ckc::format_error("unmodelled scale shape for '%s'", atom->op_id));
        const int groups = k / block;
        const int per_lane = mfma ? 1 : groups;
        const rocke_mma_scale_filter_t scales = {"e8m0", "e8m0", atom->srcs[0].scale_block_size};
        const auto* profile = rocke_mma_catalog_op_for_shape(
            &target->mma, atom->family, "fp4", "fp4", "fp32", m, m, k, &scales);
        for(int i = 0; i < 2; ++i)
        {
            const auto* layout = rocke_mma_op_src_scale_layout(atom, i, b);
            if(!profile || !layout || layout != rocke_mma_op_src_scale_layout(profile, i, b)
               || layout->frag_len != per_lane)
                throw ckc::NotImplError(
                    ckc::format_error("unmodelled scale_src%d map for '%s'", i, atom->op_id));
        }
        int shape[] = {m, groups};
        rocke_h_row_t rows[] = {{&shape[0], 1}, {&shape[1], 1}};
        int replication[] = {wave / m};
        int p_major[] = {mfma ? 2 : 0, 1}, p_minor[] = {0, 0};
        rocke_p_seq_t partitions[] = {{p_major, p_minor, 2}};
        int y_major[] = {2}, y_minor[] = {0};
        const auto* encoding = rocke_make_tile_distribution_encoding(b,
                                                                     wmma ? replication : NULL,
                                                                     wmma ? 1 : 0,
                                                                     rows,
                                                                     2,
                                                                     partitions,
                                                                     1,
                                                                     wmma ? y_major : NULL,
                                                                     wmma ? y_minor : NULL,
                                                                     wmma ? 1 : 0);
        return encoding ? rocke_h_make_tile_desc(b, out, shape, 2, encoding) : ROCKE_ERR_VALUE;
    });
}

rocke_status_t rocke_h_make_fragment(rocke_ir_builder_t* b,
                                     rocke_tiling_fragment_t* out,
                                     const rocke_tiling_tile_desc_t* tile,
                                     const rocke_type_t* dtype,
                                     rocke_value_t* value)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!out || !tile || !dtype)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid fragment arguments");
        const auto* storage = rocke_storage_vector_type(b, dtype, tile->register_count);
        if(!storage)
            return ROCKE_ERR_VALUE;
        if(value && !rocke_type_eq(value->type, storage))
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "Fragment value does not match its live storage type");
        out->tile = *tile;
        out->dtype = dtype;
        out->value = value;
        return ROCKE_OK;
    });
}

rocke_status_t
    rocke_h_fill_fragment(rocke_ir_builder_t* b, rocke_tiling_fragment_t* fragment, int scalar)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(scalar != 0)
            throw ckc::NotImplError(
                ckc::format_error("fill_fragment supports only a compile-time zero fill -- got %d; "
                                  "nonzero/runtime scalar splat is not implemented",
                                  scalar));
        if(!fragment)
            ckc::raise_status(ROCKE_ERR_VALUE, "invalid fragment arguments");
        const auto* storage
            = rocke_storage_vector_type(b, fragment->dtype, fragment->tile.register_count);
        if(!storage)
            return ROCKE_ERR_VALUE;
        fragment->value = rocke_b_zero_vec(b, storage->elem, storage->count);
        return ROCKE_OK;
    });
}
