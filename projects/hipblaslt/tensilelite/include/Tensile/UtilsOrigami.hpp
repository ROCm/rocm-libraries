/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include <cstddef>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <origami/origami.hpp>
#include <rocisa/include/enum.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{

    inline origami::data_type_t datatypeToAnalyticalDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return origami::data_type_t::Float;
        case rocisa::DataType::Double:
            return origami::data_type_t::Double;
        case rocisa::DataType::ComplexFloat:
            return origami::data_type_t::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return origami::data_type_t::ComplexDouble;
        case rocisa::DataType::Half:
            return origami::data_type_t::Half;
        case rocisa::DataType::Int8x4:
            return origami::data_type_t::Int8x4;
        case rocisa::DataType::Int32:
            return origami::data_type_t::Int32;
        case rocisa::DataType::BFloat16:
            return origami::data_type_t::BFloat16;
        case rocisa::DataType::Int8:
            return origami::data_type_t::Int8;
        case rocisa::DataType::Int64:
            return origami::data_type_t::Int64;
        case rocisa::DataType::XFloat32:
            return origami::data_type_t::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return origami::data_type_t::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return origami::data_type_t::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return origami::data_type_t::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return origami::data_type_t::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return origami::data_type_t::Float8;
        case rocisa::DataType::BFloat8:
            return origami::data_type_t::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return origami::data_type_t::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return origami::data_type_t::BFloat8Float8;
        case rocisa::DataType::Float6:
            return origami::data_type_t::Float6;
        case rocisa::DataType::BFloat6:
            return origami::data_type_t::BFloat6;   
        case rocisa::DataType::Float4:
            return origami::data_type_t::Float4;
        case rocisa::DataType::Int4:
            return origami::data_type_t::Int4;

        default:
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(type)));
        }
    }
    /**
     * The Origami config of a solution: what Origami ranks the solution with. Whoever asks
     * Origami about a solution builds its config here, so that a launch decision (the Stream-K
     * grid) is taken with the config that was ranked. The workspace is unlimited, as for
     * ranking; a caller that knows the launch's workspace overrides workspace_size and
     * workspace_size_per_elem_c.
     *
     * @param solution The solution (its sizeMapping is read)
     * @param index    Position of the config in its library's list
     */
    template <typename Solution>
    origami::config_t makeOrigamiConfig(const Solution& solution, std::size_t index = 0)
    {
        const auto& sm = solution.sizeMapping;
        // A dot2 kernel has no matrix instruction: Origami gets its vector lane widths.
        origami::dim3_t mi = {1, 1, 64};
        if(sm.matrixInstruction[0] != 0 || sm.matrixInstruction[1] != 0
           || sm.matrixInstruction[2] != 0)
            mi = {static_cast<size_t>(sm.matrixInstruction[0]),
                  static_cast<size_t>(sm.matrixInstruction[1]),
                  static_cast<size_t>(sm.matrixInstruction[2])};
        return origami::config_t{
            .mt                        = {sm.macroTile.x, sm.macroTile.y, sm.depthU},
            .mi                        = mi,
            .hand_optimized_main_loop  = (sm.customMainLoopScheduling > 0) ? true : false,
            .subtile                   = sm.useSubtileImpl,
            .occupancy                 = std::max(sm.CUOccupancy, static_cast<int>(1)),
            .workgroup_mapping         = sm.workGroupMapping,
            .cache_hints_a             = sm.cacheHintA(),
            .cache_hints_b             = sm.cacheHintB(),
            .cache_hints_d             = sm.cacheHintD(),
            .workspace_size            = std::numeric_limits<size_t>::max(),
            .workspace_size_per_elem_c = std::numeric_limits<size_t>::max(),
            .stream_k                  = (sm.isStreamK() ? (sm.hasDynamicAssignment() ? 4 : sm.hasHybridAssignment() ? 5 : 3) : 0),
            .index                     = index,
            .grvw_a                    = static_cast<std::size_t>(sm.grvwA),
            .grvw_b                    = static_cast<std::size_t>(sm.grvwB),
            .gwvw_d                    = static_cast<std::size_t>(sm.gwvwD),
            .vector_width_a            = sm.VectorWidthA,
            .vector_width_b            = sm.VectorWidthB,
            .cluster_dim               = {sm.clusterDim.x, sm.clusterDim.y, sm.clusterDim.z},
            .backend                   = origami::tensile_params_t{
                .local_split_u               = sm.LocalSplitU,
                .direct_to_vgpr_a            = sm.DirectToVgprA,
                .direct_to_vgpr_b            = sm.DirectToVgprB,
                .direct_to_lds_a             = sm.DirectToLdsA,
                .direct_to_lds_b             = sm.DirectToLdsB,
                .wavefront_size              = sm.WavefrontSize,
                .wave_group_m                = sm.waveGroup[0],
                .wave_group_n                = sm.waveGroup[1],
                .prefetch_global_read        = sm.PrefetchGlobalRead,
                .buffer_load                 = sm.BufferLoad,
                .buffer_store                = sm.BufferStore,
                .workgroup_mapping_xcc       = sm.workGroupMappingXCC,
                .workgroup_mapping_xcc_group = sm.workGroupMappingXCCGroup,
                .one_lds_buffer              = sm.OneLDSBuffer,
                .source_swap                 = sm.SourceSwap,
                .unroll_major_lds_a          = sm.UnrollMajorLDSA,
                .unroll_major_lds_b          = sm.UnrollMajorLDSB,
                .lds_transpose_read_a        = sm.enableLDSTrA,
                .lds_transpose_read_b        = sm.enableLDSTrB,
                .tdm_inst                    = sm.TDMInst,
                .prefetch_gl2                = sm.PrefetchGL2,
                .prefetch_local_read         = sm.PrefetchLocalRead,
                .half_plr                    = sm.HalfPLR,
                .multicast                   = sm.Multicast != 0,
                .adaptive_gemm               = sm.adaptiveGemm,
                .temporal_hint_a             = sm.hasTemporalHint ? sm.temporalHintA : -1,
                .temporal_hint_b             = sm.hasTemporalHint ? sm.temporalHintB : -1,
                .temporal_hint_d             = sm.hasTemporalHint ? sm.temporalHintD : -1,
                .tdm_fuse                    = sm.TDMFuse,
            },
        };
    }
} // namespace TensileLite

