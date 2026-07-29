// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <origami/types.hpp>

#include <algorithm>
#include <limits>

namespace TensileLite
{
    inline origami::config_t makeOrigamiConfig(ContractionSolution const& solution,
                                               size_t localIndex = 0)
    {
        auto const& mapping = solution.sizeMapping;
        origami::dim3_t mi;
        if(mapping.matrixInstruction[0] == 0 && mapping.matrixInstruction[1] == 0
           && mapping.matrixInstruction[2] == 0)
            mi = {1, 1, 64};
        else
            mi = {static_cast<size_t>(mapping.matrixInstruction[0]),
                  static_cast<size_t>(mapping.matrixInstruction[1]),
                  static_cast<size_t>(mapping.matrixInstruction[2])};

        origami::config_t config{
            .mt = {mapping.macroTile.x, mapping.macroTile.y, mapping.depthU},
            .mi = mi,
            .hand_optimized_main_loop = mapping.customMainLoopScheduling > 0,
            .subtile = mapping.useSubtileImpl,
            .occupancy = std::max(mapping.CUOccupancy, 1),
            .workgroup_mapping = mapping.workGroupMapping,
            .cache_hints_a = mapping.nonTemporalA,
            .cache_hints_b = mapping.nonTemporalB,
            .workspace_size = std::numeric_limits<size_t>::max(),
            .workspace_size_per_elem_c = mapping.workspaceSizePerElemC,
            .index = localIndex,
            .grvw_a = mapping.grvwA,
            .grvw_b = mapping.grvwB,
            .gwvw_d = mapping.gwvwD,
            .vector_width_a = mapping.VectorWidthA,
            .vector_width_b = mapping.VectorWidthB,
        };

        auto& tensile = config.tensile();
        tensile.depth_u = mapping.depthU;
        tensile.global_split_u = mapping.globalSplitU;
        tensile.global_accumulation = mapping.globalAccumulation;
        tensile.local_split_u = mapping.LocalSplitU;
        tensile.direct_to_vgpr_a = mapping.DirectToVgprA;
        tensile.direct_to_vgpr_b = mapping.DirectToVgprB;
        tensile.direct_to_lds_a = mapping.DirectToLdsA;
        tensile.direct_to_lds_b = mapping.DirectToLdsB;
        tensile.num_loads_coalesced_a = mapping.NumLoadsCoalescedA;
        tensile.num_loads_coalesced_b = mapping.NumLoadsCoalescedB;
        tensile.wave_num = mapping.waveNum;
        tensile.wave_group_m = mapping.waveGroup[0];
        tensile.wave_group_n = mapping.waveGroup[1];
        tensile.prefetch_global_read = mapping.PrefetchGlobalRead;
        tensile.math_clocks_unrolled_loop = mapping.MathClocksUnrolledLoop;
        tensile.workgroup_mapping_xcc = mapping.workGroupMappingXCC;
        tensile.workgroup_mapping_xcc_group = mapping.workGroupMappingXCCGroup;
        tensile.global_split_u_coalesced = mapping.globalSplitUCoalesced;
        tensile.global_split_u_wgm_round_robin = mapping.globalSplitUWorkGroupMappingRoundRobin;
        tensile.stream_k = mapping.streamK;
        tensile.stream_k_force_dp_only = mapping.streamKForceDPOnly != 0;
        tensile.schedule_iter_alg = mapping.scheduleIterAlg;
        tensile.prefetch_local_read = mapping.prefetchLocalRead;
        tensile.one_lds_buffer = mapping.oneLDSBuffer;
        tensile.transpose_lds = mapping.transposeLDS;
        tensile.source_swap = mapping.sourceSwap;
        tensile.local_read_vector_width = mapping.localReadVectorWidth;
        tensile.stagger_u = mapping.staggerU;
        tensile.stagger_u_mapping = mapping.staggerUMapping;
        tensile.lds_bytes = mapping.ldsBytes;
        tensile.total_vgprs = mapping.totalVgprs;
        tensile.accumulator_vgprs = mapping.accumulatorVgprs;
        tensile.total_sgprs = mapping.totalSgprs;
        tensile.scratch_bytes = mapping.scratchBytes;
        tensile.threads_per_workgroup = static_cast<int>(mapping.workGroupSize.x
                                                          * mapping.workGroupSize.y
                                                          * mapping.workGroupSize.z);
        tensile.compiled_cu_occupancy = mapping.CUOccupancy;
        return config;
    }
} // namespace TensileLite
