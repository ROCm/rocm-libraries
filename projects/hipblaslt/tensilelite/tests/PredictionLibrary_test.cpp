/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Host-only tests for ProblemPredictionLibrary deserialization: the Prediction
// node stores a table of solution indices, then MappingTraits copies each
// resolved ContractionSolution's SizeMapping into an aligned origami::config_t.
// These tests pin clusterDim x/y/z -> origami cluster_dim m/n/k, including the
// default {1,1,1} when SizeMapping never sets the field.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

#if defined(TENSILELITE_HAS_TILEWRIGHT)
#include <Tensile/TilewrightRanker.hpp>
#endif

#include "FallbackTestUtils.hpp"

#if defined(TENSILE_MSGPACK)
#include <Tensile/msgpack/MessagePack.hpp>
#include <msgpack.hpp>
#elif defined(TENSILE_YAML)
#include <Tensile/llvm/YAML.hpp>
#else
#error "PredictionLibrary_test requires TENSILE_MSGPACK or TENSILE_YAML"
#endif

using namespace TensileLite;

namespace
{
    std::shared_ptr<ContractionSolution> makeMappedSolution(int index)
    {
        auto solution            = std::make_shared<ContractionSolution>();
        solution->index          = index;
        solution->kernelName     = "cluster-dim-probe";
        solution->sizeMapping.macroTile         = TensileLite::dim3(256, 256, 1);
        solution->sizeMapping.depthU            = 64;
        solution->sizeMapping.matrixInstruction = {16, 16, 16, 1};
        solution->sizeMapping.CUOccupancy       = 4;
        solution->sizeMapping.workGroupMapping  = 1;
        return solution;
    }

    std::shared_ptr<ContractionSolution> makeMappedSolution(int index, TensileLite::dim3 clusterDim)
    {
        auto solution                    = makeMappedSolution(index);
        solution->sizeMapping.clusterDim = clusterDim;
        return solution;
    }

    void expectClusterDim(origami::config_t const& cfg, TensileLite::dim3 const& clusterDim)
    {
        // Tensile SizeMapping is {x,y,z}; Origami dim3_t is {m,n,k}.
        EXPECT_EQ(cfg.cluster_dim.m, clusterDim.x);
        EXPECT_EQ(cfg.cluster_dim.n, clusterDim.y);
        EXPECT_EQ(cfg.cluster_dim.k, clusterDim.z);
    }

    // Fills `lib` in place: ProblemPredictionLibrary is not copyable (std::atomic).
    void loadPredictionLibrary(std::vector<int> const&              table,
                               SolutionMap<ContractionSolution>&    solutions,
                               ContractionProblemPredictionLibrary& lib)
    {
        LibraryIOContext<ContractionSolution> ctx{"", {}, &solutions};

#if defined(TENSILE_MSGPACK)
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, std::map<std::string, std::vector<int>>{{"table", table}});
        auto handle = msgpack::unpack(buffer.data(), buffer.size());

        Serialization::MessagePackInput input(handle.get(), &ctx);
        input.input(lib);

        std::string errors;
        for(auto const& err : input.error)
        {
            if(!errors.empty())
                errors += "; ";
            errors += err;
        }
        EXPECT_TRUE(input.error.empty()) << errors;
#elif defined(TENSILE_YAML)
        std::ostringstream yaml;
        yaml << "table: [";
        for(size_t i = 0; i < table.size(); ++i)
        {
            if(i != 0)
                yaml << ", ";
            yaml << table[i];
        }
        yaml << "]\n";

        llvm::yaml::Input yin(llvm::StringRef(yaml.str()), &ctx);
        yin >> lib;
        EXPECT_FALSE(yin.error()) << yin.error().message();
#endif
    }

    hip::HipAMDGPU makeGfx950Device()
    {
        using arch_t = origami::hardware_t::architecture_t;
        hip::HipAMDGPU device;
        device.processor        = AMDGPU::Processor::gfx950;
        device.computeUnitCount = 256;
        device.analyticalHardware
            = std::make_shared<origami::hardware_t>(arch_t::gfx950,
                                                    256,
                                                    163840,
                                                    262144,
                                                    8,
                                                    1.0,
                                                    1.0,
                                                    1.0,
                                                    4000000,
                                                    1.2,
                                                    1,
                                                    std::make_tuple(0.0, 0.008, 0.0));
        return device;
    }
}

TEST(PredictionLibraryTest, CopiesClusterDimIntoOrigamiConfig)
{
    auto solution = makeMappedSolution(42, TensileLite::dim3(2, 4, 1));

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.solution_list.size(), 1u);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.solution_list[0].first, 42);
    EXPECT_EQ(lib.solution_list[0].second, solution);

    expectClusterDim(lib.origami_config_list[0], solution->sizeMapping.clusterDim);
    EXPECT_EQ(lib.origami_config_list[0].cluster_dim, (origami::dim3_t{2, 4, 1}));
}

TEST(PredictionLibraryTest, DefaultClusterDimWhenUnset)
{
    auto solution = makeMappedSolution(42);
    EXPECT_EQ(solution->sizeMapping.clusterDim.x, 1u);
    EXPECT_EQ(solution->sizeMapping.clusterDim.y, 1u);
    EXPECT_EQ(solution->sizeMapping.clusterDim.z, 1u);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    expectClusterDim(lib.origami_config_list[0], TensileLite::dim3(1, 1, 1));
}

TEST(PredictionLibraryTest, ClusterDimAxesAreNotSwapped)
{
    auto solution = makeMappedSolution(42, TensileLite::dim3(2, 1, 1));

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);

    auto const& cfg = lib.origami_config_list[0];
    expectClusterDim(cfg, TensileLite::dim3(2, 1, 1));
    EXPECT_NE(cfg.cluster_dim, (origami::dim3_t{1, 2, 1}));
}

TEST(PredictionLibraryTest, ClusterDimStaysIndexAlignedWithSolutionList)
{
    auto clustered = makeMappedSolution(42, TensileLite::dim3(2, 4, 1));
    auto plain     = makeMappedSolution(7);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, clustered);
    solutions.emplace(7, plain);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42, 7}, solutions, lib);
    ASSERT_EQ(lib.solution_list.size(), 2u);
    ASSERT_EQ(lib.origami_config_list.size(), 2u);

    EXPECT_EQ(lib.solution_list[0].first, 42);
    EXPECT_EQ(lib.solution_list[1].first, 7);
    expectClusterDim(lib.origami_config_list[0], TensileLite::dim3(2, 4, 1));
    expectClusterDim(lib.origami_config_list[1], TensileLite::dim3(1, 1, 1));
}

TEST(PredictionLibraryTest, ZeroRequestedSolutionsReturnsNone)
{
    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    ASSERT_EQ(lib.findTopSolutions(problem, device, 1).size(), 1u);
    EXPECT_TRUE(lib.findTopSolutions(problem, device, 0).empty());
    EXPECT_FALSE(lib.lastFindTopAlreadyRetAll());
}

#if defined(TENSILELITE_HAS_TILEWRIGHT)
namespace
{
    class ScratchDir
    {
    public:
        explicit ScratchDir(std::string const& name)
            : m_path(std::filesystem::temp_directory_path()
                     / (name + "_"
                        + std::to_string(
                            std::chrono::steady_clock::now().time_since_epoch().count())))
        {
            std::filesystem::create_directories(m_path);
        }

        ~ScratchDir()
        {
            std::error_code ec;
            std::filesystem::remove_all(m_path, ec);
        }

        std::filesystem::path const& path() const
        {
            return m_path;
        }

    private:
        std::filesystem::path m_path;
    };

    // Sets TENSILE_STREAMK5_FORCE_MODE (unsets it for nullptr) for the
    // lifetime of the object.
    class ScopedStreamK5ForceMode
    {
    public:
        explicit ScopedStreamK5ForceMode(char const* value)
        {
            if(char const* previous = std::getenv("TENSILE_STREAMK5_FORCE_MODE"))
                m_previous = previous;
            if(value)
                setenv("TENSILE_STREAMK5_FORCE_MODE", value, 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedStreamK5ForceMode()
        {
            if(m_previous)
                setenv("TENSILE_STREAMK5_FORCE_MODE", m_previous->c_str(), 1);
            else
                unsetenv("TENSILE_STREAMK5_FORCE_MODE");
            Debug::Instance().reloadDebugBitsForTest();
        }

    private:
        std::optional<std::string> m_previous;
    };

    std::optional<std::int64_t> findAttribute(tilewright::Config const& config,
                                              std::string const&        name)
    {
        for(auto const& attribute : config.attributes)
        {
            if(attribute.name == name)
                return attribute.value;
        }
        return std::nullopt;
    }

    // Every attribute source holds a value no other one holds, except the
    // booleans, whose A/B pairs differ.
    SizeMapping attributeProbe()
    {
        SizeMapping s{};
        s.waveNum                                = 101;
        s.workGroupSize                          = TensileLite::dim3(102, 103, 104);
        s.threadTile                             = TensileLite::dim3(105, 106, 107);
        s.gwvwC                                  = 108;
        s.staggerU                               = 109;
        s.staggerUMapping                        = 110;
        s.globalSplitUPGR                        = 111;
        s.globalSplitU                           = 112;
        s.staggerStrideShift                     = 113;
        s.workGroupMapping                       = 114;
        s.workGroupMappingXCC                    = 115;
        s.workGroupMappingXCCGroup               = 116;
        s.globalSplitUCoalesced                  = true;
        s.globalSplitUWorkGroupMappingRoundRobin = false;
        s.packBatchDims                          = 117;
        s.packSummationDims                      = 118;
        s.magicDivAlg                            = 119;
        s.streamKAtomic                          = 120;
        s.tileProcessingStrategy                 = TileProcessingStrategy::StreamK;
        s.workAssignment                         = WorkAssignment::DynamicWorkQueue;
        s.prefetchAcrossPersistent               = 121;
        s.persistentKernel                       = 122;
        s.persistentKernelAlongBatch             = true;
        s.sourceKernel                           = false;
        s.globalAccumulation                     = 123;
        s.adaptiveGemmGSUA                       = 124;
        s.activationFused                        = false;
        s.PrefetchGlobalRead                     = 125;
        s.MathClocksUnrolledLoop                 = 126;
        s.nonTemporalA                           = 127;
        s.nonTemporalB                           = 128;
        s.temporalHintA                          = 129;
        s.temporalHintB                          = 130;
        s.hasTemporalHint                        = true;
        s.adaptiveGemmNTAB                       = 131;
        s.customMainLoopScheduling               = 132;
        s.useSubtileImpl                         = true;
        s.SourceSwap                             = false;
        s.NonTemporalD                           = 133;
        s.WaveSeparateGlobalReadA                = 134;
        s.WaveSeparateGlobalReadB                = 135;
        s.UnrollLoopSwapGlobalReadOrder          = 136;
        s.DirectToVgprA                          = true;
        s.DirectToVgprB                          = false;
        s.NumLoadsCoalescedA                     = 137;
        s.NumLoadsCoalescedB                     = 138;
        s.waveGroup                              = {139, 140};
        s.VectorWidthA                           = 141;
        s.VectorWidthB                           = 142;
        s.LocalSplitU                            = 143;
        s.DirectToLdsA                           = false;
        s.DirectToLdsB                           = true;
        s.expertSchedulingMode                   = 144;
        s.clusterDim                             = TensileLite::dim3(145, 146, 147);
        return s;
    }
}

TEST(PredictionLibraryTest, TilewrightIsSilentlyUnusedWithoutAnIndex)
{
    ScratchDir dir("tensilelite_tilewright_no_index");

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    ::testing::internal::CaptureStderr();
    lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx950.dat").string());
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
}

TEST(PredictionLibraryTest, TilewrightWarnsAndIsUnusedWhenTheIndexedModelCannotLoad)
{
    ScratchDir dir("tensilelite_tilewright_bad_model");
    std::ofstream(dir.path() / "tilewright_index")
        << "TensileLibrary_Probe_gfx950\tmissing.tilewright.bin\n";

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, makeMappedSolution(42));
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);

    ::testing::internal::CaptureStderr();
    lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx950.dat").string());
    std::string const warning = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(lib.tilewright_ranker, nullptr);
    EXPECT_NE(warning.find("TensileLibrary_Probe_gfx950"), std::string::npos) << warning;
    EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
}

TEST(PredictionLibraryTest, TilewrightConfigKeepsTheCoreKernelFields)
{
    SizeMapping sizeMapping{};
    sizeMapping.macroTile         = TensileLite::dim3(256, 128, 1);
    sizeMapping.depthU            = 64;
    sizeMapping.matrixInstruction = {16, 16, 32, 1};
    sizeMapping.CUOccupancy       = 3;
    sizeMapping.nonTemporalA      = 4;
    sizeMapping.grvwA             = 8;
    sizeMapping.grvwB             = 4;
    sizeMapping.gwvwD             = 2;

    auto const config = TilewrightRanker::makeConfig(sizeMapping, 7);
    EXPECT_EQ(config.mt.m, 256u);
    EXPECT_EQ(config.mt.n, 128u);
    EXPECT_EQ(config.mt.k, 64u);
    EXPECT_EQ(config.mi.m, 16u);
    EXPECT_EQ(config.mi.n, 16u);
    EXPECT_EQ(config.mi.k, 32u);
    EXPECT_EQ(config.occupancy, 3);
    EXPECT_EQ(config.cache_hints_a, 4);
    EXPECT_EQ(config.cache_hints_b, 0);
    EXPECT_EQ(config.grvw_a, 8u);
    EXPECT_EQ(config.grvw_b, 4u);
    EXPECT_EQ(config.gwvw_d, 2u);
    EXPECT_EQ(config.index, 7u);

    sizeMapping.matrixInstruction = {0, 0, 0, 0};
    sizeMapping.CUOccupancy       = -1;
    sizeMapping.hasTemporalHint   = true;
    sizeMapping.temporalHintA     = 2;
    sizeMapping.temporalHintB     = 3;

    auto const dot2 = TilewrightRanker::makeConfig(sizeMapping, 0);
    EXPECT_EQ(dot2.mi.m, 1u);
    EXPECT_EQ(dot2.mi.n, 1u);
    EXPECT_EQ(dot2.mi.k, 64u);
    EXPECT_EQ(dot2.occupancy, 1);
    EXPECT_EQ(dot2.cache_hints_a, 0);
    EXPECT_EQ(dot2.cache_hints_b, 4);
}

TEST(PredictionLibraryTest, TilewrightConfigCarriesEveryKernelAttributeOnce)
{
    std::map<std::string, std::int64_t> const expected = {
        {"wave_num", 101},
        {"work_group_x", 102},
        {"work_group_y", 103},
        {"work_group_z", 104},
        {"thread_tile_x", 105},
        {"thread_tile_y", 106},
        {"thread_tile_z", 107},
        {"gwvw_c", 108},
        {"stagger_u", 109},
        {"stagger_u_mapping", 110},
        {"global_split_u_pgr", 111},
        {"global_split_u", 112},
        {"stagger_stride_shift", 113},
        {"workgroup_mapping", 114},
        {"workgroup_mapping_xcc", 115},
        {"workgroup_mapping_xcc_group", 116},
        {"global_split_u_coalesced", 1},
        {"global_split_u_wgm_round_robin", 0},
        {"pack_batch_dims", 117},
        {"pack_summation_dims", 118},
        {"magic_div_alg", 119},
        {"stream_k_atomic", 120},
        {"tile_processing_strategy", 2},
        {"work_assignment", 1},
        {"prefetch_across_persistent", 121},
        {"persistent_kernel", 122},
        {"persistent_kernel_along_batch", 1},
        {"source_kernel", 0},
        {"global_accumulation", 123},
        {"adaptive_gemm_gsua", 124},
        {"activation_fused", 0},
        {"prefetch_global_read", 125},
        {"math_clocks_unrolled_loop", 126},
        {"non_temporal_a", 127},
        {"non_temporal_b", 128},
        {"temporal_hint_a", 129},
        {"temporal_hint_b", 130},
        {"has_temporal_hint", 1},
        {"adaptive_gemm_ntab", 131},
        {"custom_main_loop_scheduling", 132},
        {"use_subtile_impl", 1},
        {"source_swap", 0},
        {"non_temporal_d", 133},
        {"wave_separate_global_read_a", 134},
        {"wave_separate_global_read_b", 135},
        {"unroll_loop_swap_global_read_order", 136},
        {"direct_to_vgpr_a", 1},
        {"direct_to_vgpr_b", 0},
        {"num_loads_coalesced_a", 137},
        {"num_loads_coalesced_b", 138},
        {"wave_group_0", 139},
        {"wave_group_1", 140},
        {"vector_width_a", 141},
        {"vector_width_b", 142},
        {"local_split_u", 143},
        {"direct_to_lds_a", 0},
        {"direct_to_lds_b", 1},
        {"expert_scheduling_mode", 144},
        {"cluster_dim_x", 145},
        {"cluster_dim_y", 146},
        {"cluster_dim_z", 147},
    };

    auto const config = TilewrightRanker::makeConfig(attributeProbe(), 0);

    std::map<std::string, std::vector<std::int64_t>> values;
    for(auto const& attribute : config.attributes)
        values[attribute.name].push_back(attribute.value);

    EXPECT_EQ(values.size(), expected.size());
    for(auto const& [name, value] : expected)
    {
        auto const found = values.find(name);
        if(found == values.end())
            ADD_FAILURE() << "no attribute " << name;
        else
            EXPECT_EQ(found->second, std::vector<std::int64_t>{value}) << name;
    }
}

TEST(PredictionLibraryTest, TilewrightAttributesCodeTheExecutionPolicy)
{
    SizeMapping sizeMapping{};
    for(auto const& [strategy, code] : std::vector<std::pair<TileProcessingStrategy, std::int64_t>>{
            {TileProcessingStrategy::None, 0},
            {TileProcessingStrategy::DataParallel, 1},
            {TileProcessingStrategy::StreamK, 2},
        })
    {
        sizeMapping.tileProcessingStrategy = strategy;
        EXPECT_EQ(
            findAttribute(TilewrightRanker::makeConfig(sizeMapping, 0), "tile_processing_strategy"),
            code)
            << toString(strategy);
    }
    for(auto const& [assignment, code] : std::vector<std::pair<WorkAssignment, std::int64_t>>{
            {WorkAssignment::StaticGrid, 0},
            {WorkAssignment::DynamicWorkQueue, 1},
            {WorkAssignment::Hybrid, 2},
        })
    {
        sizeMapping.workAssignment = assignment;
        EXPECT_EQ(findAttribute(TilewrightRanker::makeConfig(sizeMapping, 0), "work_assignment"),
                  code)
            << toString(assignment);
    }
}

TEST(PredictionLibraryTest, TilewrightAttributesOfAMinimalSolution)
{
    auto const config = TilewrightRanker::makeConfig(makeMappedSolution(42)->sizeMapping, 0);
    EXPECT_EQ(findAttribute(config, "workgroup_mapping"), 1);
    EXPECT_EQ(findAttribute(config, "magic_div_alg"), 1);
    EXPECT_EQ(findAttribute(config, "activation_fused"), 1);
    EXPECT_EQ(findAttribute(config, "prefetch_global_read"), 2);
    EXPECT_EQ(findAttribute(config, "local_split_u"), 1);
    EXPECT_EQ(findAttribute(config, "vector_width_a"), 1);
    EXPECT_EQ(findAttribute(config, "vector_width_b"), 1);
    EXPECT_EQ(findAttribute(config, "tile_processing_strategy"), 0);
    EXPECT_EQ(findAttribute(config, "work_assignment"), 0);
    EXPECT_EQ(findAttribute(config, "stream_k_atomic"), 0);
    EXPECT_EQ(findAttribute(config, "cluster_dim_x"), 1);
    EXPECT_EQ(findAttribute(config, "cluster_dim_y"), 1);
    EXPECT_EQ(findAttribute(config, "cluster_dim_z"), 1);
}

TEST(PredictionLibraryTest, TilewrightContextCarriesTheCuBudget)
{
    auto problem = TensileLite::testing::dummyProblem();
    EXPECT_EQ(TilewrightRanker::executionContext(problem).cu_budget, 0u);

    problem.setParams().setSmCountTarget(128);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).cu_budget, 128u);
}

TEST(PredictionLibraryTest, TilewrightContextFollowsStreamKTileScheduling)
{
    ScopedStreamK5ForceMode respectApi(nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Default);

    problem.setParams().setStreamKTileSchedulingMode(1);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Dynamic);

    problem.setParams().setStreamKTileSchedulingMode(2);
    EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule, tilewright::Schedule::Auto);
}

TEST(PredictionLibraryTest, TilewrightContextHonorsTheStreamK5ForceMode)
{
    auto problem = TensileLite::testing::dummyProblem();
    {
        ScopedStreamK5ForceMode forceDynamic("1");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
        problem.setParams().setStreamKTileSchedulingMode(2);
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
    }
    {
        ScopedStreamK5ForceMode forceStatic("0");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Default);
        problem.setParams().setStreamKTileSchedulingMode(1);
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Default);
    }
    {
        ScopedStreamK5ForceMode respectApi("-1");
        EXPECT_EQ(TilewrightRanker::executionContext(problem).schedule,
                  tilewright::Schedule::Dynamic);
    }
}

#if defined(TILEWRIGHT_TEST_WEIGHTS_DIR)
namespace
{
    std::filesystem::path shippedGfx950Model()
    {
        return std::filesystem::path(TILEWRIGHT_TEST_WEIGHTS_DIR) / "gfx950" / "gfx950"
               / "gfx950_Cijk_Ailk_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.bin";
    }

    // Loads `table` and the model at `model` as the library of the probe stem.
    void loadWithTilewright(ScratchDir const&                    dir,
                            std::filesystem::path const&         model,
                            std::vector<int> const&              table,
                            SolutionMap<ContractionSolution>&    solutions,
                            ContractionProblemPredictionLibrary& lib)
    {
        std::filesystem::copy_file(model, dir.path() / "probe.tilewright.bin");
        std::ofstream(dir.path() / "tilewright_index")
            << "TensileLibrary_Probe_gfx950\tprobe.tilewright.bin\n";
        loadPredictionLibrary(table, solutions, lib);
        lib.loadTilewright((dir.path() / "TensileLibrary_Probe_gfx950.dat").string());
    }

    std::shared_ptr<ContractionSolution> makeDot2Solution(int index)
    {
        auto solution                           = makeMappedSolution(index);
        solution->sizeMapping.matrixInstruction = {0, 0, 0, 0};
        return solution;
    }

    std::vector<int> const contestedTable = {13, 11, 12};

    // The shipped model and origami order the blocky and the skinny tile
    // differently at 1024^3; neither ranks the Dot2 kernel there.
    void addContestedKernels(SolutionMap<ContractionSolution>& solutions)
    {
        auto blocky                   = makeMappedSolution(11);
        blocky->sizeMapping.macroTile = TensileLite::dim3(128, 64, 1);
        auto skinny                   = makeMappedSolution(12);
        skinny->sizeMapping.macroTile = TensileLite::dim3(16, 256, 1);

        solutions.emplace(11, blocky);
        solutions.emplace(12, skinny);
        solutions.emplace(13, makeDot2Solution(13));
    }

    std::vector<int> solutionIndices(SolutionVector<ContractionSolution> const& solutions)
    {
        std::vector<int> indices;
        for(auto const& solution : solutions)
            indices.push_back(solution->index);
        return indices;
    }

    // The kernels the model scores for dummyProblem() on the whole device, best first.
    std::vector<int> tilewrightOrder(ContractionProblemPredictionLibrary const& lib)
    {
        tilewright::Problem const problem{
            .size     = {1024, 1024, 1024},
            .batch    = 1,
            .a_dtype  = tilewright::DataType::Float,
            .b_dtype  = tilewright::DataType::Float,
            .c_dtype  = tilewright::DataType::Float,
            .d_dtype  = tilewright::DataType::Float,
            .mi_dtype = tilewright::DataType::Float,
        };
        std::vector<int> order;
        for(auto const& r :
            lib.tilewright_ranker->candidates().rank(problem, {256, 163840, 4000000}, 3))
        {
            if(r.scored)
                order.push_back(lib.solution_list[r.config_index].first);
        }
        return order;
    }

    // What the contested library returns for `problem` without a model.
    std::vector<int> origamiOrder(ContractionProblemGemm const& problem)
    {
        SolutionMap<ContractionSolution> solutions;
        addContestedKernels(solutions);
        ContractionProblemPredictionLibrary lib;
        loadPredictionLibrary(contestedTable, solutions, lib);
        return solutionIndices(lib.findTopSolutions(problem, makeGfx950Device(), 3));
    }

    void expectOrigamiOrder(ContractionProblemPredictionLibrary const& lib,
                            ContractionProblemGemm const&              problem)
    {
        auto const device = makeGfx950Device();
        EXPECT_TRUE(lib.tilewright_ranker->rank(problem, *device.analyticalHardware, 3).empty());

        auto const origami = origamiOrder(problem);
        EXPECT_NE(origami, tilewrightOrder(lib));
        EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), origami);
    }

    void expectTilewrightOrder(ContractionProblemPredictionLibrary const& lib,
                               ContractionProblemGemm const&              problem)
    {
        auto const device = makeGfx950Device();
        EXPECT_NE(origamiOrder(problem), tilewrightOrder(lib));
        EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), tilewrightOrder(lib));
    }
}

TEST(PredictionLibraryTest, TilewrightOrdersTheKernelsItScores)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_rank");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const expected = tilewrightOrder(lib);
    ASSERT_EQ(expected.size(), 2u);

    auto const problem = TensileLite::testing::dummyProblem();
    auto const device  = makeGfx950Device();
    EXPECT_NE(origamiOrder(problem), expected);
    EXPECT_EQ(solutionIndices(lib.findTopSolutions(problem, device, 3)), expected);
    EXPECT_TRUE(lib.lastFindTopAlreadyRetAll());

    auto const best = lib.findTopSolutions(problem, device, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0]->index, expected[0]);
    EXPECT_FALSE(lib.lastFindTopAlreadyRetAll());
}

TEST(PredictionLibraryTest, TilewrightPoolPositionsAreSolutionListPositions)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_pool");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const& configs = lib.tilewright_ranker->candidates().configs();
    auto const  names   = tilewright::attribute_names(lib.tilewright_ranker->candidates().model());
    ASSERT_EQ(configs.size(), lib.solution_list.size());
    for(size_t i = 0; i < configs.size(); ++i)
    {
        auto const expected
            = TilewrightRanker::makeConfig(lib.solution_list[i].second->sizeMapping, i, &names);
        EXPECT_EQ(configs[i].index, i);
        EXPECT_EQ(configs[i].mt.m, expected.mt.m) << i;
        EXPECT_EQ(configs[i].mt.n, expected.mt.n) << i;
        EXPECT_EQ(configs[i].mi.m, expected.mi.m) << i;
        ASSERT_EQ(configs[i].attributes.size(), expected.attributes.size()) << i;
        for(size_t a = 0; a < expected.attributes.size(); ++a)
        {
            EXPECT_EQ(configs[i].attributes[a].name, expected.attributes[a].name) << i;
            EXPECT_EQ(configs[i].attributes[a].value, expected.attributes[a].value) << i;
        }
    }
}

TEST(PredictionLibraryTest, OrigamiRanksWhenTilewrightScoresNothing)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(13, makeDot2Solution(13));

    ScratchDir                          dir("tensilelite_tilewright_fallback");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), {13}, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto const top
        = lib.findTopSolutions(TensileLite::testing::dummyProblem(), makeGfx950Device(), 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0]->index, 13);
}

TEST(PredictionLibraryTest, OrigamiRanksCallsWithACuBudgetBelowTheDevice)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_cu_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(224);
    expectOrigamiOrder(lib, problem);
}

TEST(PredictionLibraryTest, TilewrightRanksCallsWithACuBudgetOfTheWholeDevice)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_full_budget");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    problem.setParams().setSmCountTarget(256);
    expectTilewrightOrder(lib, problem);
}

TEST(PredictionLibraryTest, OrigamiRanksCallsWithDynamicOrAutoStreamKScheduling)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    ScopedStreamK5ForceMode respectApi(nullptr);

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_streamk_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    for(int mode : {1, 2})
    {
        SCOPED_TRACE("streamKTileSchedulingMode " + std::to_string(mode));
        auto problem = TensileLite::testing::dummyProblem();
        problem.setParams().setStreamKTileSchedulingMode(mode);
        expectOrigamiOrder(lib, problem);
    }
}

TEST(PredictionLibraryTest, StreamK5ForceModeDecidesBetweenTilewrightAndOrigami)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    SolutionMap<ContractionSolution> solutions;
    addContestedKernels(solutions);

    ScratchDir                          dir("tensilelite_tilewright_force_mode");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), contestedTable, solutions, lib);
    ASSERT_NE(lib.tilewright_ranker, nullptr);

    auto problem = TensileLite::testing::dummyProblem();
    {
        ScopedStreamK5ForceMode forceDynamic("1");
        expectOrigamiOrder(lib, problem);
    }
    problem.setParams().setStreamKTileSchedulingMode(1);
    {
        ScopedStreamK5ForceMode forceStatic("0");
        expectTilewrightOrder(lib, problem);
    }
}
#endif
#endif
