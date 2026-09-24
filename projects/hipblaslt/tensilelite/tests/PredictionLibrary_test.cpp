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
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipHardware.hpp>

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

TEST(PredictionLibraryTest, DeviceScalarAlphaFiltersUnadvertisedSolutions)
{
    // Exercise ranking without querying a physical GPU.
    hip::HipAMDGPU hardware;
    hardware.processor = AMDGPU::Processor::gfx950;
    hardware.computeUnitCount = 256;
    hardware.deviceName = "device-alpha-selection-test";
    hardware.analyticalHardware = std::make_shared<origami::hardware_t>(
        origami::hardware_t::architecture_t::gfx950,
        256, 163840, 262144, 8, 1.0, 1.0, 1.0, 4000000, 1.2, 1,
        std::make_tuple(0.0, 0.008, 0.0));

    auto legacy = makeMappedSolution(0);
    auto scalar = makeMappedSolution(1);
    legacy->problemType.useScaleAlphaVec = 1;
    scalar->problemType.useScaleAlphaVec = 1;
    scalar->internalArgsSupport.deviceScalarAlpha = true;
    SolutionMap<ContractionSolution> solutions{{0, legacy}, {1, scalar}};
    ContractionProblemPredictionLibrary lib;
    loadPredictionLibrary({0, 1}, solutions, lib);
    auto problem = ContractionProblemGemm::GEMM(false, false, 256, 256, 64,
                                               256, 64, 256, 1.0, false, 1);
    problem.setUseScaleAlphaVec(1);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    ASSERT_EQ(lib.findTopSolutions(problem, hardware, 2).size(), 2u);

    problem.setParams().setDeviceScalarAlpha(true);
    auto selected = lib.findTopSolutions(problem, hardware, 2);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), scalar);
    scalar->problemType.useScaleAlphaVec = 0;
    EXPECT_TRUE(lib.findTopSolutions(problem, hardware, 2).empty());
}

TEST(PredictionLibraryTest, MlpDeviceScalarAlphaFiltersUnadvertisedSolutions)
{
    const AMDGPU hardware(AMDGPU::Processor::gfx900, 64, "selection test");
    auto legacy = makeMappedSolution(0);
    auto scalar = makeMappedSolution(1);
    legacy->libraryLogicIndex = 0;
    scalar->libraryLogicIndex = 1;
    legacy->problemType.useScaleAlphaVec = 1;
    scalar->problemType.useScaleAlphaVec = 1;
    scalar->internalArgsSupport.deviceScalarAlpha = true;
    MLPClassificationLibrary<ContractionProblemGemm> lib;
    lib.solutionmap = {{0, legacy}, {1, scalar}};
    lib.probFeatures = {std::make_shared<MLFeatures::FreeSizeA>(),
                        std::make_shared<MLFeatures::FreeSizeB>(),
                        std::make_shared<MLFeatures::BatchSize>(),
                        std::make_shared<MLFeatures::BoundSize>()};
    lib.model = std::make_shared<MLPClassification::MLPNet>();
    lib.model->scaler.mean.assign(12, 0.0f);
    lib.model->scaler.scale.assign(12, 1.0f);
    // Rank the unsupported solution first, then fall through to the capable one.
    lib.model->dense = MLPClassification::DenseLayer(std::vector<float>(24, 0.0f), {2.0f, 1.0f});
    auto problem = ContractionProblemGemm::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);
    problem.setUseScaleAlphaVec(1);
    ASSERT_EQ(lib.findTopSolutions(problem, hardware, 2).size(), 2u);

    problem.setParams().setDeviceScalarAlpha(true);
    auto selected = lib.findTopSolutions(problem, hardware, 2);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected.front(), scalar);
    scalar->problemType.useScaleAlphaVec = 0;
    EXPECT_TRUE(lib.findTopSolutions(problem, hardware, 2).empty());
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
