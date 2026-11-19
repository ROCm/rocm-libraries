// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Bounded global fragment verbs; mirrors global paths in tiling/emit.py. */
#include "rocke/helper_rocke.helpers.tiling.h"

#include <limits.h>
#include <string.h>

#include "rocke/dtypes.h"
#include "rocke/error_boundary.hpp"
#include "rocke/ir_storage.h"
#include "rocke/storage_internal.h"
#include "rocke/tiling_internal.h"
#include "rocke/tiling_transport_internal.h"

struct access_info
{
    int lead;
    int bits;
    int group;
    int group_words;
    int unit_bytes;
    int width;
    int step;
    int storage_count;
    bool packed;
    bool in_bounds;
    const rocke_type_t* unit;
};

static int gcd(int a, int c)
{
    while(c)
    {
        int next = a % c;
        a = c;
        c = next;
    }
    return a;
}

static int cap_alignment(int derived, int stated)
{
    return stated && stated < derived ? stated : derived;
}

static rocke_value_t*
    unit_index(rocke_ir_builder_t* b, rocke_value_t* index, const access_info& access)
{
    if(access.group == 1)
        return index;
    auto* groups = rocke_b_div(b, index, rocke_b_const_i32(b, access.group));
    return access.group_words == 1
               ? groups
               : rocke_b_mul(b, groups, rocke_b_const_i32(b, access.group_words));
}

static access_info prepare_access(rocke_ir_builder_t* b,
                                  rocke_value_t* ptr,
                                  const rocke_tiling_window_t* window,
                                  const rocke_tiling_tile_desc_t* tile,
                                  int alignment,
                                  bool packed_aligned,
                                  const char* verb,
                                  bool store)
{
    if(alignment < 0 || (alignment && (alignment & (alignment - 1))))
        ckc::raise_status(ROCKE_ERR_VALUE, "alignment_bytes must be a positive power of two");
    if(!ptr || !window || !window->tensor || !tile || !tile->layout)
        ckc::raise_status(ROCKE_ERR_VALUE, "null fragment transport argument");
    if(ptr->type->kind == ROCKE_TYPE_SMEM && alignment)
        ckc::raise_status(ROCKE_ERR_VALUE,
                          "alignment_bytes applies to global fragment transport only");
    if(ptr->type->kind != ROCKE_TYPE_PTR)
        ckc::raise_status(ROCKE_ERR_NOTIMPL,
                          "native fragment transport supports global pointers only");
    const auto* tensor = window->tensor;
    if(tensor->rank < 1 || tensor->rank > ROCKE_TILING_MAX_RANK || tile->rank < 1
       || tile->rank > ROCKE_TILING_MAX_RANK || tile->rank != tile->layout->num_X)
        ckc::raise_status(ROCKE_ERR_VALUE, "invalid fragment transport rank");
    if(tile->register_count != rocke_i_tiling_register_count(tile->layout, tile->shape))
        ckc::raise_status(ROCKE_ERR_VALUE, "tile register count does not match its distribution");
    if(tile->rank > tensor->rank)
        throw ckc::ValueError(
            ckc::format_error("%s: the tile has rank %d but the window has rank %d -- a "
                              "rank-t tile covers the window's last t axes, so t <= r",
                              verb,
                              tile->rank,
                              tensor->rank));
    access_info access = {};
    access.lead = tensor->rank - tile->rank;
    for(int axis = access.lead; axis < tensor->rank; ++axis)
        if(!tensor->has_length[axis])
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "unknown tensor lengths are supported on leading axes only");
    for(int axis = 0; axis < access.lead; ++axis)
        if(window->has_bound[axis])
            throw ckc::ValueError(
                ckc::format_error("%s: the window bounds axis %d, a leading axis of this rank-%d "
                                  "tile -- a leading axis is placed by its origin and is never "
                                  "clipped; pass None for it and keep the origin inside the tensor "
                                  "(guard the index before the access if it can fall outside)",
                                  verb,
                                  axis,
                                  tile->rank));
    const auto* info = tensor->dtype ? rocke_dtype_info(tensor->dtype->name) : NULL;
    access.unit = rocke_storage_element_type(tensor->dtype);
    if(!info || !access.unit)
        ckc::raise_status(ROCKE_ERR_VALUE, "invalid fragment storage dtype");
    access.bits = info->encoded_bits;
    access.packed = access.bits < 8;
    const int common = gcd(access.bits, 32);
    access.group = access.packed ? 32 / common : 1;
    access.group_words = access.packed ? access.bits / common : 1;
    access.unit_bytes = rocke_dtype_info(access.unit->name)->encoded_bits / 8;
    const auto* storage = rocke_storage_vector_type(b, tensor->dtype, tile->register_count);
    if(!storage)
        ckc::raise_status(ROCKE_ERR_VALUE, b->err);
    access.storage_count = storage->count;
    if(packed_aligned && !access.packed)
        throw ckc::ValueError(ckc::format_error(
            "%s failed - packed_runs_aligned=True with %s, expected a packed "
            "sub-byte dtype (fp4, fp6, i4); %s runtime origins and bounds need no "
            "statement, so drop the argument",
            verb,
            info->name,
            info->name));
    const auto* layout = tile->layout;
    int run_axis = -1, run_length = 1;
    if(layout->num_Y)
    {
        const int major = layout->Ys_major[layout->num_Y - 1];
        const int minor = layout->Ys_minor[layout->num_Y - 1];
        if(major && tensor->strides[access.lead + major - 1] == 1)
        {
            int64_t inner = 1;
            for(int i = minor + 1; i < layout->Hs[major - 1].count; ++i)
                inner *= layout->Hs[major - 1].levels[i];
            if(inner == 1)
            {
                run_axis = access.lead + major - 1;
                run_length = layout->Hs[major - 1].levels[minor];
            }
        }
    }
    access.in_bounds = true;
    for(int axis = 0; axis < tile->rank; ++axis)
    {
        if(tile->shape[axis] <= 0)
            ckc::raise_status(ROCKE_ERR_VALUE, "tile shape must be positive");
        access.in_bounds
            &= !rocke_i_tiling_may_overhang(window, access.lead + axis, tile->shape[axis]);
    }
    if(access.packed)
    {
        const auto need = ckc::format_error(
            "%s of %s needs each lane's registers to form runs of whole 32-bit words "
            "along the tensor's stride-1 axis (multiples of %d elements, starting on %s "
            "%d-element boundary)",
            verb,
            info->name,
            access.group,
            access.group == 8 ? "an" : "a",
            access.group);
        if(run_axis < 0 || run_length % access.group)
            throw ckc::ValueError(
                need
                + ckc::format_error("; this layout gives runs of %d %s -- choose a layout whose "
                                    "innermost register run lies along the stride-1 axis",
                                    run_length,
                                    run_length == 1 ? "element" : "elements"));
        for(int axis = 0; axis < tensor->rank; ++axis)
            if(axis != run_axis && tensor->strides[axis] % access.group)
                throw ckc::ValueError(
                    need
                    + ckc::format_error(
                        "; tensor axis %d steps by %d elements, so its runs start off "
                        "that boundary -- pad that stride to a multiple of %d",
                        axis,
                        tensor->strides[axis],
                        access.group));
        for(int which = 0; which < 2; ++which)
        {
            const auto index
                = which ? rocke_i_tiling_bound(window, run_axis) : window->origin[run_axis];
            int64_t constant;
            if(!rocke_i_tiling_constant_index(index, &constant))
            {
                if(!packed_aligned)
                    throw ckc::ValueError(ckc::format_error(
                        "%s: the window %s on the stride-1 axis is a runtime value, "
                        "which %s cannot check is a multiple of %d elements (whole "
                        "32-bit words of %s) -- pass packed_runs_aligned=True if it "
                        "always is",
                        verb,
                        which ? "bound" : "origin",
                        verb,
                        access.group,
                        info->name));
            }
            else if(constant % access.group)
            {
                if(which)
                    throw ckc::ValueError(ckc::format_error(
                        "%s of a clipped %s window needs its bound along the stride-1 "
                        "axis to be a multiple of %d elements (whole 32-bit words) "
                        "-- got %lld",
                        verb,
                        info->name,
                        access.group,
                        (long long)constant));
                throw ckc::ValueError(
                    need
                    + ckc::format_error("; the window origin on the stride-1 axis is %lld -- place "
                                        "the window on a multiple of %d",
                                        (long long)constant,
                                        access.group));
            }
        }
    }
    access.width = access.packed ? access.group_words : 1;
    if(!store && run_axis >= 0 && access.in_bounds)
    {
        if(access.packed)
        {
            const int64_t run_words = rocke_i_elements_to_storage(access.bits, run_length);
            const int widths[] = {4, 3, 2, 1};
            for(int width : widths)
            {
                bool divides = width % access.group_words == 0 && run_words % width == 0;
                for(int axis = 0; axis < tensor->rank; ++axis)
                    if(axis != run_axis)
                        divides &= rocke_i_elements_to_storage(access.bits, tensor->strides[axis])
                                       % width
                                   == 0;
                if(divides)
                {
                    access.width = width;
                    break;
                }
            }
        }
        else
        {
            const char* unit = access.unit->name;
            bool wide = strcmp(unit, "f16") == 0 || strcmp(unit, "bf16") == 0
                        || strcmp(unit, "f32") == 0 || strcmp(unit, "i32") == 0
                        || strcmp(unit, "i8") == 0 || strcmp(unit, "fp8e4m3") == 0
                        || strcmp(unit, "bf8e5m2") == 0;
            const int widths[] = {16, 8, 4, 2};
            for(int width : widths)
            {
                bool divides = wide && width <= 16 / access.unit_bytes && run_length % width == 0;
                for(int axis = 0; axis < tensor->rank; ++axis)
                    if(axis != run_axis)
                        divides &= tensor->strides[axis] % width == 0;
                if(divides)
                {
                    access.width = width;
                    break;
                }
            }
        }
    }
    if(!store && alignment && access.packed && access.group_words == 3 && access.in_bounds
       && run_axis >= 0)
    {
        int64_t origin;
        const int paired_words = access.group_words * 2;
        bool bundle = rocke_i_tiling_constant_index(window->origin[run_axis], &origin)
                      && origin % (access.group * 2) == 0
                      && rocke_i_elements_to_storage(access.bits, run_length) % paired_words == 0;
        for(int axis = 0; axis < tensor->rank; ++axis)
            if(axis != run_axis)
                bundle &= rocke_i_elements_to_storage(access.bits, tensor->strides[axis])
                              % paired_words
                          == 0;
        if(bundle)
            access.width = paired_words;
    }
    access.step = access.packed ? access.width * access.group / access.group_words : access.width;
    return access;
}

rocke_status_t rocke_h_load_fragment(rocke_ir_builder_t* b,
                                     rocke_tiling_fragment_t* out,
                                     rocke_value_t* ptr,
                                     const rocke_tiling_window_t* window,
                                     const rocke_tiling_tile_desc_t* tile,
                                     rocke_value_t* thread,
                                     int alignment_bytes,
                                     bool packed_runs_aligned)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!out || !thread)
            ckc::raise_status(ROCKE_ERR_VALUE, "null fragment transport argument");
        const auto access = prepare_access(
            b, ptr, window, tile, alignment_bytes, packed_runs_aligned, "load_fragment", false);
        auto* value = rocke_b_zero_vec(b, access.unit, access.storage_count);
        rocke_value_t* rebased = NULL;
        int64_t constant_origin = 0;
        if(rocke_i_tiling_can_rebase_global(ptr, window, tile, alignment_bytes, access.in_bounds))
        {
            rocke_tiling_window_t runtime_window;
            rocke_i_tiling_split_window(window, &runtime_window, &constant_origin);
            rocke_value_t* coords[ROCKE_TILING_MAX_RANK];
            rocke_value_t* positions[ROCKE_TILING_MAX_RANK];
            rocke_i_tiling_coordinates(b, tile, thread, 0, coords);
            rocke_i_tiling_positions(b, &runtime_window, tile->rank, coords, positions);
            auto* offset = rocke_i_tiling_byte_address(b, &runtime_window, tile->rank, positions);
            rebased = rocke_b_global_ptr_add(b, ptr, rocke_b_sext(b, offset, rocke_i64()));
        }
        rocke_value_t* zero = NULL;
        for(int reg = 0; reg < tile->register_count; reg += access.step)
        {
            rocke_value_t* address;
            rocke_value_t* mask = NULL;
            auto* load_ptr = rebased ? rebased : ptr;
            if(rebased)
            {
                int64_t deltas[ROCKE_TILING_MAX_RANK];
                rocke_i_tiling_register_offsets(tile, reg, deltas);
                address = rocke_b_const_i32(
                    b, rocke_i_tiling_storage_displacement(window, tile, deltas, constant_origin));
            }
            else
            {
                rocke_value_t* coords[ROCKE_TILING_MAX_RANK];
                rocke_value_t* positions[ROCKE_TILING_MAX_RANK];
                rocke_i_tiling_coordinates(b, tile, thread, reg, coords);
                rocke_i_tiling_positions(b, window, tile->rank, coords, positions);
                address = unit_index(
                    b, rocke_i_tiling_address(b, window, tile->rank, positions), access);
                mask = rocke_i_tiling_clip_mask(b, window, tile, positions);
            }
            const int first_slot = int(rocke_i_elements_to_storage(access.bits, reg));
            if(!mask && access.width > 1)
            {
                rocke_value_t* pieces[16];
                int piece_count;
                const int derived_align = access.unit_bytes * (access.width & -access.width);
                if(rocke_h_load_storage_chunks(b,
                                               load_ptr,
                                               address,
                                               access.unit,
                                               access.width,
                                               cap_alignment(derived_align, alignment_bytes),
                                               false,
                                               rocke_type_eq(access.unit, rocke_i32()),
                                               pieces,
                                               16,
                                               &piece_count)
                   != ROCKE_OK)
                    return ROCKE_ERR_VALUE;
                auto* loaded = pieces[0];
                for(int i = 1; i < piece_count; ++i)
                    loaded = rocke_b_vec_concat(b, loaded, pieces[i]);
                for(int i = 0; i < access.width; ++i)
                    value = rocke_b_vec_insert(
                        b, value, rocke_b_vec_extract(b, loaded, i), first_slot + i);
                continue;
            }
            if(mask && !zero)
                zero = rocke_b_vec_extract(b, rocke_b_zero_vec(b, access.unit, 1), 0);
            for(int i = 0; i < access.width; ++i)
            {
                auto* at = i ? rocke_b_add(b, address, rocke_b_const_i32(b, i)) : address;
                const int align = cap_alignment(access.unit_bytes, alignment_bytes);
                auto* loaded = mask ? rocke_b_masked_global_load(
                                          b, load_ptr, at, mask, zero, access.unit, align)
                                    : rocke_b_global_load(b, load_ptr, at, access.unit, align);
                value = rocke_b_vec_insert(b, value, loaded, first_slot + i);
            }
        }
        return rocke_h_make_fragment(b, out, tile, window->tensor->dtype, value);
    });
}

rocke_status_t rocke_h_store_fragment(rocke_ir_builder_t* b,
                                      rocke_value_t* ptr,
                                      const rocke_tiling_window_t* window,
                                      const rocke_tiling_fragment_t* fragment,
                                      rocke_value_t* thread,
                                      int alignment_bytes,
                                      bool packed_runs_aligned)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        if(!rocke_i_live(b))
            return ROCKE_ERR_VALUE;
        if(!thread)
            ckc::raise_status(ROCKE_ERR_VALUE, "null fragment transport argument");
        if(!fragment || !fragment->value)
            ckc::raise_status(ROCKE_ERR_VALUE, "Fragment has no materialized value");
        const auto* tile = &fragment->tile;
        const auto access = prepare_access(
            b, ptr, window, tile, alignment_bytes, packed_runs_aligned, "store_fragment", true);
        const auto* src_storage
            = rocke_storage_vector_type(b, fragment->dtype, tile->register_count);
        if(!src_storage || !rocke_type_eq(src_storage, fragment->value->type))
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "Fragment value does not match its live storage type");
        bool identity = rocke_type_eq(fragment->dtype, window->tensor->dtype);
        if(!identity && !rocke_type_eq(fragment->dtype, rocke_f32()))
            throw ckc::NotImplError(
                ckc::format_error("unsupported fragment cast on store -- src='%s', dst='%s', "
                                  "expected src==dst or src=='f32'",
                                  fragment->dtype->name,
                                  window->tensor->dtype->name));
        rocke_value_t* coords[ROCKE_TILING_MAX_RANK];
        rocke_value_t* base_positions[ROCKE_TILING_MAX_RANK];
        rocke_i_tiling_coordinates(b, tile, thread, 0, coords);
        rocke_i_tiling_positions(b, window, tile->rank, coords, base_positions);
        auto* base_index
            = unit_index(b, rocke_i_tiling_address(b, window, tile->rank, base_positions), access);
        const int align = cap_alignment(access.unit_bytes, alignment_bytes);
        for(int reg = 0; reg < tile->register_count; reg += access.step)
        {
            int64_t deltas[ROCKE_TILING_MAX_RANK];
            rocke_value_t* positions[ROCKE_TILING_MAX_RANK];
            rocke_i_tiling_register_offsets(tile, reg, deltas);
            for(int axis = 0; axis < tile->rank; ++axis)
            {
                positions[axis]
                    = deltas[axis]
                          ? rocke_b_add(b, base_positions[axis], rocke_b_const_i32(b, deltas[axis]))
                          : base_positions[axis];
            }
            const int64_t unit_offset
                = rocke_i_tiling_storage_displacement(window, tile, deltas, 0);
            auto* address = unit_offset
                                ? rocke_b_add(b, base_index, rocke_b_const_i32(b, unit_offset))
                                : base_index;
            rocke_value_t* units[3];
            const int first_slot = int(rocke_i_elements_to_storage(access.bits, reg));
            for(int i = 0; i < access.width; ++i)
            {
                auto* value = rocke_b_vec_extract(b, fragment->value, first_slot + i);
                units[i] = identity ? value : rocke_b_cast_f32_to(b, value, window->tensor->dtype);
            }
            auto* mask = rocke_i_tiling_clip_mask(b, window, tile, positions);
            if(mask)
            {
                auto branch = rocke_b_scf_if(b, mask);
                rocke_b_region_enter(b, branch.then_region);
            }
            for(int i = 0; i < access.width; ++i)
            {
                auto* at = i ? rocke_b_add(b, address, rocke_b_const_i32(b, i)) : address;
                rocke_b_global_store(b, ptr, at, units[i], align);
            }
            if(mask)
                rocke_b_region_leave(b);
        }
        return ROCKE_OK;
    });
}
