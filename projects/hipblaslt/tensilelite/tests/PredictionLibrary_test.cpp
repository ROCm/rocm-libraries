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
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

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
    EXPECT_EQ(lib.tilewright_candidates, nullptr);
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
    EXPECT_EQ(lib.tilewright_candidates, nullptr);
    EXPECT_NE(warning.find("TensileLibrary_Probe_gfx950"), std::string::npos) << warning;
    EXPECT_NE(warning.find("ranking with origami"), std::string::npos) << warning;
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
}

TEST(PredictionLibraryTest, TilewrightOrdersTheKernelsItScores)
{
    if(!std::filesystem::exists(shippedGfx950Model()))
        GTEST_SKIP() << "no tilewright model at " << shippedGfx950Model();

    // The shipped model and origami order these two tiles differently at 1024^3.
    auto blocky                   = makeMappedSolution(11);
    blocky->sizeMapping.macroTile = TensileLite::dim3(128, 64, 1);
    auto skinny                   = makeMappedSolution(12);
    skinny->sizeMapping.macroTile = TensileLite::dim3(16, 256, 1);

    SolutionMap<ContractionSolution> solutions;
    solutions.emplace(11, blocky);
    solutions.emplace(12, skinny);
    solutions.emplace(13, makeDot2Solution(13));

    ScratchDir                          dir("tensilelite_tilewright_rank");
    ContractionProblemPredictionLibrary lib;
    loadWithTilewright(dir, shippedGfx950Model(), {13, 11, 12}, solutions, lib);
    ASSERT_NE(lib.tilewright_candidates, nullptr);

    tilewright::Problem const tilewrightProblem{
        .size     = {1024, 1024, 1024},
        .batch    = 1,
        .a_dtype  = tilewright::DataType::Float,
        .b_dtype  = tilewright::DataType::Float,
        .c_dtype  = tilewright::DataType::Float,
        .d_dtype  = tilewright::DataType::Float,
        .mi_dtype = tilewright::DataType::Float,
    };
    std::vector<int> expected;
    for(auto const& r :
        lib.tilewright_candidates->rank(tilewrightProblem, {256, 163840, 4000000}, 3))
    {
        if(r.scored)
            expected.push_back(lib.solution_list[r.config_index].first);
    }
    // At M = 1024 neither tilewright nor origami ranks the Dot2 kernel.
    ASSERT_EQ(expected.size(), 2u);

    auto const       problem = TensileLite::testing::dummyProblem();
    auto const       device  = makeGfx950Device();
    std::vector<int> picked;
    for(auto const& solution : lib.findTopSolutions(problem, device, 3))
        picked.push_back(solution->index);
    EXPECT_EQ(picked, expected);
    EXPECT_TRUE(lib.lastFindTopAlreadyRetAll());

    auto const best = lib.findTopSolutions(problem, device, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0]->index, expected[0]);
    EXPECT_FALSE(lib.lastFindTopAlreadyRetAll());
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
    ASSERT_NE(lib.tilewright_candidates, nullptr);

    auto const top
        = lib.findTopSolutions(TensileLite::testing::dummyProblem(), makeGfx950Device(), 1);
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0]->index, 13);
}
#endif
#endif
