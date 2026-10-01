// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <sstream>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <Tensile/ContractionSolution.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;

namespace
{
    ContractionProblemGemm persistentProblem(size_t m = 257, size_t n = 385,
                                            size_t k = 129, size_t batches = 3)
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, std::max(size_t{1}, k), m, 1.0, false, batches);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }

    void configurePersistentSolution(ContractionSolution& solution, int outer, int layout)
    {
        solution.kernelName = "prebuilt_data_parallel_symbol";
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
        solution.sizeMapping.workAssignment = WorkAssignment::StaticGrid;
        solution.sizeMapping.macroTile = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU = 64;
        solution.sizeMapping.workGroupSize = TensileLite::dim3(256, 1, 1);
        solution.sizeMapping.matrixInstruction = {16, 16, 4, 1};
        solution.sizeMapping.globalSplitU = 0;
        solution.sizeMapping.globalAccumulation = 0;
        solution.sizeMapping.workGroupMapping = 1;
        solution.sizeMapping.workGroupMappingXCC = 1;
        solution.sizeMapping.CUOccupancy = 1;
        solution.internalArgsSupport.version = outer;
        solution.internalArgsSupport.persistentLoopArgsVersion = layout;
        solution.internalArgsSupport.useUniversalArgs = true;
    }

    AMDGPU persistentDevice(size_t grid = 7)
    {
        auto device = TensileLite::testing::makeDevice(
            TensileLite::testing::_MI350_CHIP_ID, TensileLite::testing::_CPX_CU, "mi350cpx");
        device.skDynamicGrid = 0;
        device.skFixedGrid = grid;
        return device;
    }

    ContractionInputs persistentInputs(float alpha = 2.5f)
    {
        ContractionInputs inputs;
        inputs.alpha = alpha;
        inputs.beta = -1.25f;
        return inputs;
    }

    KernelArguments::ArgPair argument(KernelArguments const& args, std::string const& name)
    {
        auto iterator = KernelArguments::const_iterator(args, name);
        if(iterator == args.end())
            throw std::runtime_error("Missing packed argument " + name);
        return *iterator;
    }

    bool hasArgument(KernelArguments const& args, std::string const& name)
    {
        return KernelArguments::const_iterator(args, name) != args.end();
    }

    size_t offset(KernelArguments const& args, std::string const& name)
    {
        return static_cast<uint8_t const*>(argument(args, name).first)
             - static_cast<uint8_t const*>(args.data());
    }

    template <typename T>
    T value(KernelArguments const& args, std::string const& name)
    {
        auto field = argument(args, name);
        EXPECT_EQ(field.second, sizeof(T)) << name;
        T result{};
        if(field.second == sizeof(T))
            std::memcpy(&result, field.first, sizeof(T));
        return result;
    }

    class PersistentArgumentLayoutTest : public ::testing::TestWithParam<int>
    {
    };
}

TEST_P(PersistentArgumentLayoutTest, PackedLayoutAndLaunch)
{
    const int outer = GetParam();
    constexpr int layout = 0;
    ContractionSolution solution;
    configurePersistentSolution(solution, outer, layout);
    auto problem = persistentProblem();
    auto device = persistentDevice();
    auto launch = solution.resolvePersistentSettings(problem, device);
    auto invocation = solution.generateSingleCall<true>(
        problem, persistentInputs(), device, launch, GSUSettings{});
    auto const& args = invocation.args;

    ASSERT_EQ(launch.totalTiles, 36u);
    EXPECT_EQ(launch.grid, 7u);
    EXPECT_EQ(launch.reduction, origami::reduction_t::none);
    EXPECT_EQ(invocation.numWorkGroups.x, 7u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    EXPECT_EQ(invocation.kernelName, "prebuilt_data_parallel_symbol");
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);

    const char* first = "itersPerTile";
    const char* grid = "skGrid";
    EXPECT_EQ(value<uint32_t>(args, first), 3u);
    EXPECT_EQ(value<uint32_t>(args, grid), 7u);
    EXPECT_EQ(offset(args, grid) - offset(args, first), 16u);
    EXPECT_FLOAT_EQ(value<float>(args, "alpha"), 2.5f);
    EXPECT_FLOAT_EQ(value<float>(args, "beta"), -1.25f);
    if(outer < 3)
        EXPECT_LT(offset(args, "beta"), offset(args, first));
    else
        EXPECT_EQ(offset(args, "alpha"), offset(args, first) + 24u);

    for(auto name : {"ws", "Flags", "AddressWS", "AddressFlags"})
        EXPECT_FALSE(hasArgument(args, name)) << name;
    for(auto name : {"magicNumberItersPerTile", "magicShiftItersPerTile", "SKItersPerWG", "skTiles"})
        EXPECT_TRUE(hasArgument(args, name)) << name;
    EXPECT_EQ(value<uint32_t>(args, "SKItersPerWG"), 0u);
    EXPECT_EQ(value<uint32_t>(args, "skTiles"), 0u);

    // The XML carries the actual packed names, offsets and widths for the
    // independent Python-signature integration comparison.
    std::ostringstream packed;
    for(auto name : {"gemm_count", "internalArgs", "internalArgs1", "internalArgs2",
                     "numWorkGroups", "size_0", "size_1", "size_2", "size_3",
                     "a", "b", "c", "d", "strideA1", "strideA2", "strideB1", "strideB2",
                     "strideC1", "strideC2", "strideD1", "strideD2", "alpha", "beta",
                     "itersPerTile", "magicNumberItersPerTile", "magicShiftItersPerTile",
                     "SKItersPerWG", "skGrid", "skTiles",
                     "batchOffsetD", "batchOffsetC", "batchOffsetA", "batchOffsetB"})
    {
        if(hasArgument(args, name))
            packed << name << ':' << offset(args, name) << ':' << argument(args, name).second << '\n';
    }
    RecordProperty("outerVersion", outer);
    RecordProperty("layoutVersion", layout);
    RecordProperty("arguments", packed.str());
    RecordProperty("argumentBytes", static_cast<int>(args.size()));
}

INSTANTIATE_TEST_SUITE_P(LegacyOuterVersions, PersistentArgumentLayoutTest,
                        ::testing::Values(0, 1, 2, 3));
