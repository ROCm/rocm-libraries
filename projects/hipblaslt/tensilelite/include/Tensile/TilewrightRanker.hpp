// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>

#include <origami/hardware.hpp>
#include <rocisa/include/enum.hpp>
#include <tilewright/model.hpp>
#include <tilewright/types.hpp>

namespace TensileLite
{
    inline tilewright::DataType datatypeToTilewrightDatatype(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return tilewright::DataType::Float;
        case rocisa::DataType::Double:
            return tilewright::DataType::Double;
        case rocisa::DataType::ComplexFloat:
            return tilewright::DataType::ComplexFloat;
        case rocisa::DataType::ComplexDouble:
            return tilewright::DataType::ComplexDouble;
        case rocisa::DataType::Half:
            return tilewright::DataType::Half;
        case rocisa::DataType::Int8x4:
            return tilewright::DataType::Int8x4;
        case rocisa::DataType::Int32:
            return tilewright::DataType::Int32;
        case rocisa::DataType::BFloat16:
            return tilewright::DataType::BFloat16;
        case rocisa::DataType::Int8:
            return tilewright::DataType::Int8;
        case rocisa::DataType::Int64:
            return tilewright::DataType::Int64;
        case rocisa::DataType::XFloat32:
            return tilewright::DataType::XFloat32;
        case rocisa::DataType::Float8_fnuz:
            return tilewright::DataType::Float8_fnuz;
        case rocisa::DataType::BFloat8_fnuz:
            return tilewright::DataType::BFloat8_fnuz;
        case rocisa::DataType::Float8BFloat8_fnuz:
            return tilewright::DataType::Float8BFloat8_fnuz;
        case rocisa::DataType::BFloat8Float8_fnuz:
            return tilewright::DataType::BFloat8Float8_fnuz;
        case rocisa::DataType::Float8:
            return tilewright::DataType::Float8;
        case rocisa::DataType::BFloat8:
            return tilewright::DataType::BFloat8;
        case rocisa::DataType::Float8BFloat8:
            return tilewright::DataType::Float8BFloat8;
        case rocisa::DataType::BFloat8Float8:
            return tilewright::DataType::BFloat8Float8;
        case rocisa::DataType::Float6:
            return tilewright::DataType::Float6;
        case rocisa::DataType::BFloat6:
            return tilewright::DataType::BFloat6;
        case rocisa::DataType::Float4:
            return tilewright::DataType::Float4;
        default:
            return tilewright::DataType::None;
        }
    }

    /**
     * Ranks the kernels of a Prediction library with the tilewright model that
     * ships for the library. Pool position i is configs[i].
     */
    class TilewrightRanker
    {
    public:
        // Throws std::invalid_argument for a null model or an invalid attribute name.
        TilewrightRanker(tilewright::ModelPtr model, std::vector<tilewright::Config> configs)
            : m_candidates(std::move(model),
                           std::move(configs),
                           tilewright::PoolOptions{.tie_break = tilewright::TieBreak::PoolOrder})
        {
        }

        // Looks up the stem of `logicFile` in the tilewright_index of its
        // directory. Returns nullptr when no model is listed for the stem, or,
        // with a warning, when the listed model cannot be loaded.
        static tilewright::ModelPtr loadModel(std::string const& logicFile)
        {
            if(logicFile.empty())
                return nullptr;

            std::string dir  = ".";
            std::string stem = logicStem(logicFile, &dir);

            std::string error;
            auto        model = tilewright::load_model_by_index(stem, dir, &error);
            if(!model && !error.empty())
                warnUnusable(stem, error);
            return model;
        }

        // A ranker over `poolSize` kernels whose config i is built from
        // sizeMapping(i). Throws what the CandidateSet throws.
        template <typename SizeMappingOf>
        static std::shared_ptr<const TilewrightRanker>
            make(tilewright::ModelPtr model, size_t poolSize, SizeMappingOf&& sizeMapping)
        {
            auto const                      names = tilewright::attribute_names(*model);
            std::vector<tilewright::Config> configs;
            configs.reserve(poolSize);
            for(size_t i = 0; i < poolSize; i++)
                configs.push_back(makeConfig(sizeMapping(i), i, &names));
            return std::make_shared<const TilewrightRanker>(std::move(model), std::move(configs));
        }

        // Looks up the stem of `logicFile` in the tilewright_index of its
        // directory; config i is built from solutions[i]. Returns nullptr when
        // no model is listed for the stem, or, with a warning, when the listed
        // model cannot be used.
        template <typename MySolution>
        static std::shared_ptr<const TilewrightRanker>
            load(std::string const&                                              logicFile,
                 std::vector<std::pair<int, std::shared_ptr<MySolution>>> const& solutions)
        {
            return load(loadModel(logicFile), logicFile, solutions);
        }

        // As above, for the model loadModel(logicFile) returned.
        template <typename MySolution>
        static std::shared_ptr<const TilewrightRanker>
            load(tilewright::ModelPtr                                            model,
                 std::string const&                                              logicFile,
                 std::vector<std::pair<int, std::shared_ptr<MySolution>>> const& solutions)
        {
            if(!model)
                return nullptr;

            std::string error;
            try
            {
                return make(
                    std::move(model), solutions.size(), [&](size_t i) -> SizeMapping const& {
                        return solutions[i].second->sizeMapping;
                    });
            }
            catch(std::exception const& e)
            {
                error = e.what();
            }
            catch(...)
            {
                error = "unknown exception";
            }
            warnUnusable(logicStem(logicFile), error);
            return nullptr;
        }

        // Whether a library ranked with `model` also ranks its problem type's
        // Equality kernels: as TENSILE_TILEWRIGHT_BANK says, or, when it is
        // unset, when the model reads kernel attributes, which the Equality
        // variants of one tile need to be told apart.
        static bool ranksBank(tilewright::Model const& model)
        {
            int const setting = Debug::Instance().tilewrightBank();
            if(setting >= 0)
                return setting == 1;
            return !tilewright::attribute_names(model).empty();
        }

        // Attaches the attributes listed in `attributeNames`, or all of them
        // when it is null.
        static tilewright::Config makeConfig(SizeMapping const&              sizeMapping,
                                             size_t                          index,
                                             std::vector<std::string> const* attributeNames
                                             = nullptr)
        {
            tilewright::Dim3 const mtSize{
                sizeMapping.macroTile.x, sizeMapping.macroTile.y, sizeMapping.depthU};

            auto const& mi = sizeMapping.matrixInstruction;

            // Same stand-in for dot2 kernels as the origami config.
            tilewright::Dim3 miSize{1, 1, 64};
            if(mi[0] != 0 || mi[1] != 0 || mi[2] != 0)
                miSize = {static_cast<size_t>(mi[0]),
                          static_cast<size_t>(mi[1]),
                          static_cast<size_t>(mi[2])};

            return {
                .mt            = mtSize,
                .mi            = miSize,
                .occupancy     = std::max(sizeMapping.CUOccupancy, 1),
                .cache_hints_a = sizeMapping.cacheHintA(),
                .cache_hints_b = sizeMapping.cacheHintB(),
                .grvw_a        = sizeMapping.grvwA,
                .grvw_b        = sizeMapping.grvwB,
                .gwvw_d        = sizeMapping.gwvwD,
                .index         = index,
                .attributes    = makeAttributes(sizeMapping, attributeNames),
            };
        }

        template <typename MyProblem>
        static tilewright::ExecutionContext executionContext(MyProblem const& problem)
        {
            int const smCountTarget = problem.getParams().smCountTarget();
            return {
                .cu_budget = smCountTarget > 0 ? static_cast<size_t>(smCountTarget) : 0,
                .schedule  = schedule(problem),
            };
        }

        // Pool positions of the kernels tilewright scores for `problem`, best
        // first. Calls outside the execution context the model was trained for
        // get none.
        template <typename MyProblem>
        std::vector<size_t> rank(MyProblem const&           problem,
                                 origami::hardware_t const& hardware,
                                 int                        numSolutions) const
        {
            size_t m     = 1;
            size_t n     = 1;
            size_t k     = 1;
            size_t batch = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
                m *= problem.freeSizeA(i);
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
                n *= problem.freeSizeB(i);
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
                k *= problem.boundSize(i);
            for(size_t i = 0; i < problem.batchIndices().size(); ++i)
                batch *= problem.batchSize(i);

            tilewright::Problem const tilewrightProblem = {
                .size  = {m, n, k},
                .batch = batch,
                .a_transpose
                = problem.transA() ? tilewright::Transpose::T : tilewright::Transpose::N,
                .b_transpose
                = problem.transB() ? tilewright::Transpose::T : tilewright::Transpose::N,
                .a_dtype  = datatypeToTilewrightDatatype(problem.a().dataType()),
                .b_dtype  = datatypeToTilewrightDatatype(problem.b().dataType()),
                .c_dtype  = datatypeToTilewrightDatatype(problem.c().dataType()),
                .d_dtype  = datatypeToTilewrightDatatype(problem.d().dataType()),
                .mi_dtype = problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                                ? tilewright::DataType::XFloat32
                                : datatypeToTilewrightDatatype(problem.computeInputTypeA()),
            };
            tilewright::Hardware const tilewrightHardware = {
                .N_CU         = hardware.N_CU,
                .lds_capacity = hardware.lds_capacity,
                .L2_capacity  = hardware.L2_capacity,
            };
            size_t const depth = numSolutions < 0 ? std::numeric_limits<size_t>::max()
                                                  : static_cast<size_t>(numSolutions);

            std::vector<size_t> positions;
            try
            {
                auto const ranked = m_candidates.rank_scored(
                    tilewrightProblem, tilewrightHardware, executionContext(problem), depth);
                positions.reserve(ranked.size());
                for(auto const& r : ranked)
                    positions.push_back(r.config_index);
            }
            catch(std::exception const&)
            {
                // Ranking only fails to allocate; origami ranks every kernel then.
                positions.clear();
            }
            return positions;
        }

        tilewright::CandidateSet const& candidates() const noexcept
        {
            return m_candidates;
        }

    private:
        // Logic stems contain no '.', and the file may carry two extensions.
        static std::string logicStem(std::string const& logicFile, std::string* dir = nullptr)
        {
            std::string stem         = logicFile;
            size_t      directoryPos = stem.rfind('/');
#ifdef _WIN32
            if(directoryPos == std::string::npos)
                directoryPos = stem.rfind('\\');
#endif
            if(directoryPos != std::string::npos)
            {
                if(dir != nullptr)
                    *dir = stem.substr(0, directoryPos);
                stem = stem.substr(directoryPos + 1);
            }
            return stem.substr(0, stem.find('.'));
        }

        static void warnUnusable(std::string const& stem, std::string const& error)
        {
            std::cerr << "hipBLASLt Warning: TENSILE_USE_TILEWRIGHT is set but the tilewright "
                         "model for "
                      << stem << " could not be used (" << error
                      << "); ranking with origami instead.\n";
        }

        // Default exactly when ExactLogicLibrary::findTopSolutions computes
        // effectiveDynamic as false.
        template <typename MyProblem>
        static tilewright::Schedule schedule(MyProblem const& problem)
        {
            switch(Debug::Instance().streamK5ForceMode())
            {
            case 0:
                return tilewright::Schedule::Default;
            case 1:
                return tilewright::Schedule::Dynamic;
            default:
                break;
            }
            switch(problem.getParams().streamKTileSchedulingMode())
            {
            case 0:
                return tilewright::Schedule::Default;
            case 2:
                return tilewright::Schedule::Auto;
            default:
                return tilewright::Schedule::Dynamic;
            }
        }

        static std::int64_t attributeValue(TileProcessingStrategy strategy)
        {
            switch(strategy)
            {
            case TileProcessingStrategy::None:
                return 0;
            case TileProcessingStrategy::DataParallel:
                return 1;
            case TileProcessingStrategy::StreamK:
                return 2;
            }
            throw std::runtime_error("Invalid TileProcessingStrategy");
        }

        static std::int64_t attributeValue(WorkAssignment assignment)
        {
            switch(assignment)
            {
            case WorkAssignment::StaticGrid:
                return 0;
            case WorkAssignment::DynamicWorkQueue:
                return 1;
            case WorkAssignment::Hybrid:
                return 2;
            }
            throw std::runtime_error("Invalid WorkAssignment");
        }

        // The training pipeline derives the same names and values from the
        // sizeMapping keys of the library .dat; both sides change together.
        static std::vector<tilewright::Attribute>
            makeAttributes(SizeMapping const& s, std::vector<std::string> const* names)
        {
            std::vector<tilewright::Attribute> attributes;
            if(names != nullptr && names->empty())
                return attributes;

            auto add = [&attributes, names](char const* name, auto value) {
                if(names == nullptr
                   || std::find(names->begin(), names->end(), name) != names->end())
                    attributes.push_back({name, static_cast<std::int64_t>(value)});
            };
            add("wave_num", s.waveNum);
            add("work_group_x", s.workGroupSize.x);
            add("work_group_y", s.workGroupSize.y);
            add("work_group_z", s.workGroupSize.z);
            add("thread_tile_x", s.threadTile.x);
            add("thread_tile_y", s.threadTile.y);
            add("thread_tile_z", s.threadTile.z);
            add("gwvw_c", s.gwvwC);
            add("stagger_u", s.staggerU);
            add("stagger_u_mapping", s.staggerUMapping);
            add("global_split_u_pgr", s.globalSplitUPGR);
            add("global_split_u", s.globalSplitU);
            add("stagger_stride_shift", s.staggerStrideShift);
            add("workgroup_mapping", s.workGroupMapping);
            add("workgroup_mapping_xcc", s.workGroupMappingXCC);
            add("workgroup_mapping_xcc_group", s.workGroupMappingXCCGroup);
            add("global_split_u_coalesced", s.globalSplitUCoalesced);
            add("global_split_u_wgm_round_robin", s.globalSplitUWorkGroupMappingRoundRobin);
            add("pack_batch_dims", s.packBatchDims);
            add("pack_summation_dims", s.packSummationDims);
            add("magic_div_alg", s.magicDivAlg);
            add("stream_k_atomic", s.streamKAtomic);
            add("tile_processing_strategy", attributeValue(s.tileProcessingStrategy));
            add("work_assignment", attributeValue(s.workAssignment));
            add("prefetch_across_persistent", s.prefetchAcrossPersistent);
            add("persistent_kernel", s.persistentKernel);
            add("persistent_kernel_along_batch", s.persistentKernelAlongBatch);
            add("source_kernel", s.sourceKernel);
            add("global_accumulation", s.globalAccumulation);
            add("adaptive_gemm_gsua", s.adaptiveGemmGSUA);
            add("activation_fused", s.activationFused);
            add("prefetch_global_read", s.PrefetchGlobalRead);
            add("math_clocks_unrolled_loop", s.MathClocksUnrolledLoop);
            add("non_temporal_a", s.nonTemporalA);
            add("non_temporal_b", s.nonTemporalB);
            add("temporal_hint_a", s.temporalHintA);
            add("temporal_hint_b", s.temporalHintB);
            add("has_temporal_hint", s.hasTemporalHint);
            add("adaptive_gemm_ntab", s.adaptiveGemmNTAB);
            add("custom_main_loop_scheduling", s.customMainLoopScheduling);
            add("use_subtile_impl", s.useSubtileImpl);
            add("source_swap", s.SourceSwap);
            add("non_temporal_d", s.NonTemporalD);
            add("wave_separate_global_read_a", s.WaveSeparateGlobalReadA);
            add("wave_separate_global_read_b", s.WaveSeparateGlobalReadB);
            add("unroll_loop_swap_global_read_order", s.UnrollLoopSwapGlobalReadOrder);
            add("direct_to_vgpr_a", s.DirectToVgprA);
            add("direct_to_vgpr_b", s.DirectToVgprB);
            add("num_loads_coalesced_a", s.NumLoadsCoalescedA);
            add("num_loads_coalesced_b", s.NumLoadsCoalescedB);
            add("wave_group_0", s.waveGroup[0]);
            add("wave_group_1", s.waveGroup[1]);
            add("vector_width_a", s.VectorWidthA);
            add("vector_width_b", s.VectorWidthB);
            add("local_split_u", s.LocalSplitU);
            add("direct_to_lds_a", s.DirectToLdsA);
            add("direct_to_lds_b", s.DirectToLdsB);
            add("expert_scheduling_mode", s.expertSchedulingMode);
            add("cluster_dim_x", s.clusterDim.x);
            add("cluster_dim_y", s.clusterDim.y);
            add("cluster_dim_z", s.clusterDim.z);
            return attributes;
        }

        tilewright::CandidateSet m_candidates;
    };

    /**
     * The pools a library ranks when tilewright also ranks its problem type's
     * Equality kernels (the bank): one per device, because a device runs only
     * the bank kernels whose hardware predicate accepts it.
     *
     * Pool position i < P (P kernels in the library's own pool) is the library's
     * kernel i; position P + j is bank[j]. The bank holds, in ascending solution
     * index, the Equality-row solutions whose hardware predicate accepts the
     * device, of every library file that holds the same problem type for the
     * same architecture (libraryProblemTypeKey; the library's own file
     * included), less the solutions of the library's own pool. The training
     * pipeline builds the same pool for its benchmark device.
     */
    template <typename MySolution>
    class TilewrightBank
    {
    public:
        struct Pool
        {
            std::vector<std::shared_ptr<MySolution>> bank;
            // Null when the bank is empty or could not be ranked: the library's
            // own ranker ranks its own pool then.
            std::shared_ptr<const TilewrightRanker> ranker;
        };

        explicit TilewrightBank(tilewright::ModelPtr model)
            : m_model(std::move(model))
        {
        }

        tilewright::ModelPtr const& model() const noexcept
        {
            return m_model;
        }

        // The pool of `device`, made by build() the first time the device asks.
        template <typename Build>
        Pool const& pool(AMDGPU const& device, Build&& build) const
        {
            Device const key{device.processor, device.computeUnitCount, device.pciChipId()};
            if(Entry const* last = m_last.load(std::memory_order_acquire);
               last && last->device == key)
                return last->pool;

            std::lock_guard<std::mutex> lock(m_mutex);
            for(auto const& entry : m_entries)
            {
                if(entry.device == key)
                {
                    m_last.store(&entry, std::memory_order_release);
                    return entry.pool;
                }
            }
            m_entries.push_back(Entry{key, build()});
            m_last.store(&m_entries.back(), std::memory_order_release);
            return m_entries.back().pool;
        }

    private:
        // The fields AMDGPU::operator== compares.
        struct Device
        {
            AMDGPU::Processor  processor;
            int                computeUnitCount;
            std::optional<int> pciChipId;

            bool operator==(Device const& other) const
            {
                return processor == other.processor && computeUnitCount == other.computeUnitCount
                       && pciChipId == other.pciChipId;
            }
        };

        struct Entry
        {
            Device device;
            Pool   pool;
        };

        tilewright::ModelPtr              m_model;
        mutable std::mutex                m_mutex;
        mutable std::list<Entry>          m_entries;
        mutable std::atomic<Entry const*> m_last{nullptr};
    };
} // namespace TensileLite
