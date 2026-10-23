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

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>

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

TEST(PredictionLibraryTest, LegacyNonTemporalBecomesOrigamiCacheHint)
{
    auto solution                      = makeMappedSolution(42);
    solution->sizeMapping.nonTemporalA = 4;
    solution->sizeMapping.nonTemporalB = 0;

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, 4);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, 0);
}

TEST(PredictionLibraryTest, TemporalHintsOverrideLegacyNonTemporal)
{
    auto solution                         = makeMappedSolution(42);
    solution->sizeMapping.hasTemporalHint = true;
    solution->sizeMapping.temporalHintA   = 1;
    solution->sizeMapping.temporalHintB   = 2;
    solution->sizeMapping.nonTemporalA    = 0;
    solution->sizeMapping.nonTemporalB    = 4;

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, solution);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 1u);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, 4);
    EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, 0);
}

// cacheHintA/B only treat temporal hints 1 (TH_NT) and 3 (TH_LU / TH_WB) as
// non-temporal. TH_NT_RT, TH_RT_NT, TH_NT_HT, and TH_NT_WB (4-7) stay cached,
// and that mapping overrides a legacy non-temporal value.
TEST(PredictionLibraryTest, UnmappedTemporalHintsStayCached)
{
    struct Case
    {
        int hint;
        int cache;
    };
    // 3 is the other non-temporal encoding. 4-7 currently collapse to 0.
    Case const cases[] = {{3, 4}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};

    for(auto const& c : cases)
    {
        auto solution                         = makeMappedSolution(42);
        solution->sizeMapping.hasTemporalHint = true;
        solution->sizeMapping.temporalHintA   = c.hint;
        solution->sizeMapping.temporalHintB   = c.hint;
        solution->sizeMapping.nonTemporalA    = 4;
        solution->sizeMapping.nonTemporalB    = 4;

        SolutionMap<ContractionSolution> solutions;
        solutions.emplace(42, solution);

        ContractionProblemPredictionLibrary lib;
        loadPredictionLibrary({42}, solutions, lib);
        ASSERT_EQ(lib.origami_config_list.size(), 1u) << "temporalHint " << c.hint;
        EXPECT_EQ(lib.origami_config_list[0].cache_hints_a, c.cache) << "temporalHint " << c.hint;
        EXPECT_EQ(lib.origami_config_list[0].cache_hints_b, c.cache) << "temporalHint " << c.hint;
    }
}

// The per-tile flags are written once while the library is loaded.
// An NT-B solution at 128x256 must not mark the 256x256 solution.
TEST(PredictionLibraryTest, NonTemporalAvailabilityIsMemoizedPerMacroTile)
{
    auto ntB                        = makeMappedSolution(42);
    ntB->sizeMapping.macroTile      = TensileLite::dim3(128, 256, 1);
    ntB->sizeMapping.nonTemporalB   = 4;
    auto temporal                   = makeMappedSolution(7);
    temporal->sizeMapping.macroTile = TensileLite::dim3(256, 256, 1);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(42, ntB);
    solutions.emplace(7, temporal);

    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42, 7}, solutions, lib);
    ASSERT_EQ(lib.origami_config_list.size(), 2u);

    EXPECT_FALSE(lib.origami_config_list[0].non_temporal_a_available);
    EXPECT_TRUE(lib.origami_config_list[0].non_temporal_b_available);
    EXPECT_FALSE(lib.origami_config_list[1].non_temporal_a_available);
    EXPECT_FALSE(lib.origami_config_list[1].non_temporal_b_available);
}

TEST(PredictionLibraryTest, SpatialPaddingChangesRankingButNotExplicitIndexLookup)
{
    auto clustered = makeMappedSolution(42, TensileLite::dim3(8, 1, 1));
    auto plain = makeMappedSolution(7);
    for(auto const& solution : {clustered, plain})
    {
        solution->sizeMapping.tileProcessingStrategy = TileProcessingStrategy::StreamK;
        solution->sizeMapping.workAssignment = WorkAssignment::StaticGrid;
        solution->hardwarePredicate = std::make_shared<Predicates::True<Hardware>>();
        solution->problemPredicate = std::make_shared<Predicates::True<ContractionProblemGemm>>();
    }
    clustered->internalArgsSupport.persistentLoopArgsVersion = 2;
    SolutionMap<ContractionSolution> solutions{{42, clustered}, {7, plain}};
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({42, 7}, solutions, lib);
    hip::HipAMDGPU device;
    device.computeUnitCount = 256;
    device.analyticalHardware = std::make_shared<origami::hardware_t>(
        origami::hardware_t::architecture_t::gfx950, 256, 163840, 262144,
        8, 1.0, 1.0, 1.0, 4000000, 1.2, 1, std::make_tuple(0.0, 0.008, 0.0));
    auto problem = ContractionProblemGemm::GEMM(
        false, false, 1, 8192, 8192, 1, 8192, 1, 1.0, false, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    const auto ranked = lib.findTopSolutions(problem, device, 2);
    ASSERT_EQ(ranked.size(), 2u);
    EXPECT_EQ(ranked[0], plain);
    EXPECT_EQ(ranked[1], clustered);
    EXPECT_EQ(lib.getSolutionByIndex(problem, device, 42), clustered);
    EXPECT_EQ(clustered->sizeMapping.clusterDim.x, 8u);
}
