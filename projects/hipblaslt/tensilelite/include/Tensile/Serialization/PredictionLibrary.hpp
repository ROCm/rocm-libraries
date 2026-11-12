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
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>

#include <Tensile/PredictionLibrary.hpp>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <tensilelitehost/export.h>
#include <iostream>
#include <unordered_map>

namespace TensileLite
{
    namespace Serialization
    {
        /// ISA major version of the processor a solution's hardware predicate requires, or 0
        /// when it names none. Processor values are gfx numbers (gfx90a = 910), so the
        /// hundreds are the major.
        template <typename MySolution>
        int predictionIsaMajor(MySolution const& solution)
        {
            auto fromGpu = [](std::shared_ptr<Predicates::Predicate<AMDGPU>> const& p) {
                std::vector<std::shared_ptr<Predicates::Predicate<AMDGPU>>> terms{p};
                if(auto all = std::dynamic_pointer_cast<Predicates::And<AMDGPU>>(p))
                    terms = all->value;
                for(auto const& term : terms)
                    if(auto eq = std::dynamic_pointer_cast<Predicates::GPU::ProcessorEqual>(term))
                        return static_cast<int>(eq->value) / 100;
                return 0;
            };
            std::vector<std::shared_ptr<Predicates::Predicate<Hardware>>> terms{
                solution.hardwarePredicate};
            if(auto all
               = std::dynamic_pointer_cast<Predicates::And<Hardware>>(solution.hardwarePredicate))
                terms = all->value;
            for(auto const& term : terms)
                if(auto gpu
                   = std::dynamic_pointer_cast<Predicates::IsSubclass<Hardware, AMDGPU>>(term))
                    if(int major = fromGpu(gpu->value))
                        return major;
            return 0;
        }

        /// Rewrites the mapping and A/B cache-hint fields of `config` to the values the kernel
        /// runs with, so solutions that run the same code get equal configs:
        ///  - cluster kernels branch over wgmXCC and DefaultWGM (WorkGroupMappingAlgos.py);
        ///  - persistent kernels without clusters on gfx12 (rocisa WorkGroupIdFromTTM) reread
        ///    WorkGroup0 from ttmp9 after the wgmXCC and PersistentXCCMapping remaps
        ///    (WorkAssignment initialize), so their WGMXCC is dead;
        ///  - WGM -1 is DefaultWGM's identity, and the launch skips WGMXCC 0 like 1;
        ///  - a shared TDM load keeps only a TH that A and B agree on.
        /// A persistent kernel's auto mapping (WGM 0 with WGMXCC -1) stays auto, for Origami to
        /// fill in as calculateAutoWGM does. Returns true when the solution declares a value
        /// its kernel ignores.
        template <typename MySizeMapping>
        bool normalizePredictionConfig(MySizeMapping const& sm,
                                       int                  isaMajor,
                                       origami::config_t&   config)
        {
            auto&      tp        = config.tensile();
            const bool clustered = sm.clusterDim.x * sm.clusterDim.y != 1;
            const bool autoWgm   = sm.isPersistent() && config.workgroup_mapping == 0
                                 && tp.workgroup_mapping_xcc == -1;
            bool ignored = false;

            if(config.workgroup_mapping == -1)
                config.workgroup_mapping = 1;
            if(tp.workgroup_mapping_xcc == 0)
                tp.workgroup_mapping_xcc = 1;
            if(clustered)
            {
                ignored = config.workgroup_mapping != 1 || tp.workgroup_mapping_xcc != 1;
                config.workgroup_mapping = 1;
                tp.workgroup_mapping_xcc = 1;
            }
            else if(isaMajor == 12 && sm.isPersistent() && !autoWgm)
            {
                ignored                  = tp.workgroup_mapping_xcc != 1;
                tp.workgroup_mapping_xcc = 1;
            }
            if(tp.workgroup_mapping_xcc == 1)
                tp.workgroup_mapping_xcc_group = -1;

            if(sm.tdmLoadSharesTemporalHint() && sm.temporalHintA != sm.temporalHintB)
                ignored = true;
            config.cache_hints_a = sm.effectiveCacheHintA();
            config.cache_hints_b = sm.effectiveCacheHintB();
            return ignored;
        }

        /// config_t::operator== leaves out fields the latency model reads.
        inline bool samePrediction(origami::config_t const& a, origami::config_t const& b)
        {
            return a == b && a.occupancy == b.occupancy && a.grid_selection == b.grid_selection
                   && a.workspace_size == b.workspace_size
                   && a.workspace_size_per_elem_c == b.workspace_size_per_elem_c;
        }

        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<ProblemPredictionLibrary<MyProblem, MySolution>, IO>
        {
            using Library = ProblemPredictionLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                auto ctx = static_cast<LibraryIOContext<MySolution>*>(iot::getContext(io));
                if(ctx == nullptr)
                {
                    iot::setError(io,
                                  "ProblemPredictionLibrary requires that context be "
                                  "set to a SolutionMap.");
                }
                std::vector<int> mappingIndices;
                if(iot::outputting(io))
                {
                    mappingIndices.reserve(lib.solution_list.size());

                    for(auto const& pair : lib.solution_list)
                        mappingIndices.push_back(pair.first);

                    iot::mapRequired(io, "table", mappingIndices);
                }
                else
                {
                    iot::mapRequired(io, "table", mappingIndices);
                    if(mappingIndices.empty())
                        iot::setError(io,
                                      "ProblemPredictionLibrary requires non empty "
                                      "mapping index set.");

                    // Effective configs, index-aligned with solution_list; deduplicated below.
                    std::vector<origami::config_t> configs;
                    std::vector<bool>              ignoresKnob;
                    for(std::size_t local_index = 0; local_index < mappingIndices.size(); local_index++)
                    {
                        int  index    = mappingIndices[local_index];
                        auto solution = resolveContextSolution(ctx, index);
                        if(!solution)
                        {
                            iot::setError(
                                io,
                                concatenate("[ProblemPredictionLibrary] Invalid solution index: ",
                                            index));
                        }
                        else
                        {
                            // configs below is built from this solution's
                            // sizeMapping and must stay index-aligned with
                            // solution_list, so both are filled here rather than
                            // deferred.
                            lib.solution_list.emplace_back(index, solution);

                            origami::dim3_t origami_mi;
                            if(solution->sizeMapping.matrixInstruction[0] == 0
                               && solution->sizeMapping.matrixInstruction[1] == 0
                               && solution->sizeMapping.matrixInstruction[2] == 0)
                            {
                                // Override dot2 instruction with vector lane widths
                                origami_mi = {1, 1, 64};
                            }
                            else
                            {
                                origami_mi = {
                                    static_cast<size_t>(solution->sizeMapping.matrixInstruction[0]),
                                    static_cast<size_t>(solution->sizeMapping.matrixInstruction[1]),
                                    static_cast<size_t>(
                                        solution->sizeMapping.matrixInstruction[2])};
                            }

                            if(Debug::Instance().printPropertyEvaluation()
                               && solution->sizeMapping.CUOccupancy <= 0)
                            {
                                std::cerr << "TensileLite::DEBUG: sizeMapping.CUOccupancy="
                                          << solution->sizeMapping.CUOccupancy
                                          << " (<=0) for solution '" << solution->kernelName
                                          << "'; clamping to 1 in origami config.\n";
                            }
                            origami::config_t origami_config = {
                                .mt = {solution->sizeMapping.macroTile.x,
                                       solution->sizeMapping.macroTile.y,
                                       solution->sizeMapping.depthU},
                                .mi = origami_mi,
                                .hand_optimized_main_loop
                                = (solution->sizeMapping.customMainLoopScheduling > 0) ? true
                                                                                       : false,
                                .subtile                   = solution->sizeMapping.useSubtileImpl,
                                .occupancy
                                = std::max(solution->sizeMapping.CUOccupancy, static_cast<int>(1)),
                                .workgroup_mapping         = solution->sizeMapping.workGroupMapping,
                                .cache_hints_a             = solution->sizeMapping.cacheHintA(),
                                .cache_hints_b             = solution->sizeMapping.cacheHintB(),
                                .cache_hints_d             = solution->sizeMapping.cacheHintD(),
                                .workspace_size            = std::numeric_limits<size_t>::max(),
                                .workspace_size_per_elem_c = std::numeric_limits<size_t>::max(),
                                .stream_k                  = (solution->sizeMapping.isStreamK() ? (solution->sizeMapping.hasDynamicAssignment() ? 4 : solution->sizeMapping.hasHybridAssignment() ? 5 : 3) : 0),
                                .index                     = local_index,
                                .grvw_a                    = static_cast<std::size_t>(
                                    solution->sizeMapping.grvwA),
                                .grvw_b                    = static_cast<std::size_t>(
                                    solution->sizeMapping.grvwB),
                                .gwvw_d                    = static_cast<std::size_t>(
                                    solution->sizeMapping.gwvwD),
                                .vector_width_a            = solution->sizeMapping.VectorWidthA,
                                .vector_width_b            = solution->sizeMapping.VectorWidthB,
                                .cluster_dim               = {solution->sizeMapping.clusterDim.x,
                                                              solution->sizeMapping.clusterDim.y,
                                                              solution->sizeMapping.clusterDim.z},
                                .backend                   = origami::tensile_params_t{
                                    .local_split_u        = solution->sizeMapping.LocalSplitU,
                                    .direct_to_vgpr_a     = solution->sizeMapping.DirectToVgprA,
                                    .direct_to_vgpr_b     = solution->sizeMapping.DirectToVgprB,
                                    .direct_to_lds_a      = solution->sizeMapping.DirectToLdsA,
                                    .direct_to_lds_b      = solution->sizeMapping.DirectToLdsB,
                                    .wavefront_size       = solution->sizeMapping.WavefrontSize,
                                    .wave_group_m         = solution->sizeMapping.waveGroup[0],
                                    .wave_group_n         = solution->sizeMapping.waveGroup[1],
                                    .prefetch_global_read = solution->sizeMapping.PrefetchGlobalRead,
                                    .buffer_load          = solution->sizeMapping.BufferLoad,
                                    .buffer_store         = solution->sizeMapping.BufferStore,
                                    .workgroup_mapping_xcc
                                    = solution->sizeMapping.workGroupMappingXCC,
                                    .workgroup_mapping_xcc_group
                                    = solution->sizeMapping.workGroupMappingXCCGroup,
                                    .one_lds_buffer       = solution->sizeMapping.OneLDSBuffer,
                                    .source_swap          = solution->sizeMapping.SourceSwap,
                                    .unroll_major_lds_a   = solution->sizeMapping.UnrollMajorLDSA,
                                    .unroll_major_lds_b   = solution->sizeMapping.UnrollMajorLDSB,
                                    .lds_transpose_read_a = solution->sizeMapping.enableLDSTrA,
                                    .lds_transpose_read_b = solution->sizeMapping.enableLDSTrB,
                                    .tdm_inst             = solution->sizeMapping.TDMInst,
                                    .prefetch_gl2         = solution->sizeMapping.PrefetchGL2,
                                    .prefetch_local_read
                                    = solution->sizeMapping.PrefetchLocalRead,
                                    .half_plr             = solution->sizeMapping.HalfPLR,
                                    .multicast = solution->sizeMapping.clusterMulticast(),
                                    .cluster_aid_mapping
                                    = solution->sizeMapping.ClusterAIDMapping,
                                    .tdm_separate_ab = solution->sizeMapping.TDMSeparateAB,
                                    .persistent = solution->sizeMapping.isPersistent(),
                                    .adaptive_gemm = solution->sizeMapping.adaptiveGemm,
                                },
                            };

                            ignoresKnob.push_back(normalizePredictionConfig(
                                solution->sizeMapping, predictionIsaMajor(*solution), origami_config));
                            configs.emplace_back(origami_config);
                        }
                    }

                    // Identical effective configs tie exactly: Origami ranks one config for
                    // all of them. Selection tries a solution whose declared values its kernel
                    // all uses first, then the lower index.
                    std::unordered_map<std::size_t, std::vector<std::size_t>> byHash;
                    for(std::size_t i = 0; i < configs.size(); ++i)
                    {
                        auto&       bucket = byHash[configs[i].hash()];
                        std::size_t group  = lib.origami_config_list.size();
                        for(std::size_t g : bucket)
                            if(samePrediction(lib.origami_config_list[g], configs[i]))
                                group = g;
                        if(group == lib.origami_config_list.size())
                        {
                            bucket.push_back(group);
                            configs[i].index = group;
                            lib.origami_config_list.push_back(configs[i]);
                            lib.config_solutions.push_back({i});
                            continue;
                        }
                        auto& members = lib.config_solutions[group];
                        auto  at      = std::find_if(members.begin(), members.end(), [&](std::size_t m) {
                            return ignoresKnob[m] && !ignoresKnob[i];
                        });
                        members.insert(at, i);
                    }
                    if(Debug::Instance().printPropertyEvaluation())
                    {
                        std::cerr << "TensileLite::DEBUG: PredictionLibrary: " << configs.size()
                                  << " solutions, " << lib.origami_config_list.size()
                                  << " distinct effective origami configs ("
                                  << configs.size() - lib.origami_config_list.size()
                                  << " duplicates dropped from ranking), "
                                  << std::count(ignoresKnob.begin(), ignoresKnob.end(), true)
                                  << " declare a value their kernel ignores.\n";
                    }
                    // Once per library load. rank_configs reads the flags.
                    origami::memoize_non_temporal_availability(lib.origami_config_list);
                }
            }
            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite

