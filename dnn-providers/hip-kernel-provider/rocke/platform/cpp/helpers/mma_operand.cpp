// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors helpers/mma_operand.py: live storage to selected instruction ABI. */
#include "rocke/helper_rocke.helpers.mma_operand.h"

#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/ir_storage.h"
#include "rocke/mfma_scale_internal.h"
#include "rocke/wmma_scale_internal.h"

rocke_value_t* rocke_h_prepare_scaled_mma_operand(rocke_ir_builder_t* b,
                                                  const rocke_mma_op_t* atom,
                                                  const char* role,
                                                  const rocke_tiling_fragment_t* fragment)
{
    return ckc::guard_builder(b, [&]() -> rocke_value_t* {
        if(!rocke_i_live(b))
            return NULL;
        if(!role)
            ckc::raise_status(ROCKE_ERR_VALUE, "scaled MMA operand role=None, expected 'a' or 'b'");
        if(strcmp(role, "a") != 0 && strcmp(role, "b") != 0)
            throw ckc::ValueError(
                ckc::format_error("scaled MMA operand role='%s', expected 'a' or 'b'", role));
        if(!fragment)
            ckc::raise_status(ROCKE_ERR_TYPE, "scaled MMA operand requires a Fragment");
        if(!atom)
            ckc::raise_status(ROCKE_ERR_TYPE, "scaled MMA operand requires an MmaOp");
        const int index = strcmp(role, "a") == 0 ? 0 : 1;
        if(!atom->family || !atom->op_id || !atom->srcs[index].dtype)
            ckc::raise_status(ROCKE_ERR_TYPE, "scaled MMA operand requires complete atom metadata");
        if(fragment->tile.rank < 1 || fragment->tile.rank > ROCKE_TILING_MAX_RANK)
            ckc::raise_status(ROCKE_ERR_NOTIMPL,
                              "native fragment encoding exceeds bounded rank/level limits");
        rocke_matrix_fragment_layout_t layout;
        if(strcmp(atom->family, "mfma_scaled") == 0 && rocke_gfx950_scaled_mfma(atom->op_id))
            layout = rocke_mfma_matrix_layout(atom->srcs[index].dtype);
        else if(strcmp(atom->family, "wmma_scaled") == 0)
        {
            const auto* operation = rocke_gfx1250_scaled_wmma(atom->op_id);
            if(!operation)
                throw ckc::ValueError(
                    ckc::format_error("unsupported scaled MMA operand ABI: %s", atom->op_id));
            (void)rocke_scaled_wmma_contract(operation);
            layout = rocke_scaled_wmma_matrix_layout(operation, index == 1);
        }
        else
            throw ckc::ValueError(
                ckc::format_error("unsupported scaled MMA operand ABI: %s", atom->op_id));
        if(!fragment->dtype || strcmp(fragment->dtype->name, atom->srcs[index].dtype) != 0)
            throw ckc::ValueError(
                ckc::format_error("scaled MMA operand %s dtype='%s', expected '%s'",
                                  role,
                                  fragment->dtype ? fragment->dtype->name : "None",
                                  atom->srcs[index].dtype));
        if(uint64_t(fragment->tile.register_count) != layout.fragment.count)
            throw ckc::ValueError(
                ckc::format_error("scaled MMA operand %s count=%d, expected %llu logical elements",
                                  role,
                                  fragment->tile.register_count,
                                  (unsigned long long)layout.fragment.count));
        if(!fragment->value)
        {
            char shape[256];
            size_t used = snprintf(shape, sizeof(shape), "(");
            for(int i = 0; i < fragment->tile.rank && used < sizeof(shape); ++i)
                used += snprintf(shape + used,
                                 sizeof(shape) - used,
                                 "%s%d",
                                 i ? ", " : "",
                                 fragment->tile.shape[i]);
            if(used < sizeof(shape))
                snprintf(shape + used, sizeof(shape) - used, fragment->tile.rank == 1 ? ",)" : ")");
            throw ckc::ValueError(ckc::format_error(
                "Fragment(shape=%s, dtype=%s) not filled -- call "
                "fill_fragment(b, fragment, 0), or use the Fragment load_fragment returns",
                shape,
                fragment->dtype->name));
        }
        const auto* storage
            = rocke_storage_vector_type(b, fragment->dtype, fragment->tile.register_count);
        if(!storage || !rocke_type_eq(storage, fragment->value->type))
            ckc::raise_status(ROCKE_ERR_VALUE,
                              "Fragment value does not match its live storage type");
        const int live = int(layout.fragment.count * layout.fragment.packing.slot_bits / 32);
        const auto* payload_type = rocke_vector_type(b, rocke_i32(), live);
        auto* value = fragment->value;
        if(!rocke_type_eq(value->type, payload_type))
            value = rocke_b_bitcast(b, value, payload_type);
        const int padding = int(layout.fragment.carrier_count) - live;
        if(padding)
            value = rocke_b_vec_concat(
                b, value, rocke_b_vector_splat(b, rocke_b_const_i32(b, 0), padding));
        return value;
    });
}
