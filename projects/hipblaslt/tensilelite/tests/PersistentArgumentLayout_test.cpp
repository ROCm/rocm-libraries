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
        device.persistentDynamicGrid = 0;
        device.persistentFixedGrid = grid;
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

    std::vector<uint8_t> bytes(KernelArguments const& args, size_t first, size_t last)
    {
        auto data = static_cast<uint8_t const*>(args.data());
        return {data + first, data + last};
    }

    class PersistentArgumentLayoutTest : public ::testing::TestWithParam<std::tuple<int, int>>
    {
    };
}

TEST_P(PersistentArgumentLayoutTest, PackedLayoutAndLaunch)
{
    auto [outer, layout] = GetParam();
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

    auto first = layout ? "ItersPerTile" : "itersPerTile";
    auto grid = layout ? "PersistentGrid" : "skGrid";
    EXPECT_EQ(value<uint32_t>(args, first), 3u);
    EXPECT_EQ(value<uint32_t>(args, grid), 7u);
    EXPECT_EQ(offset(args, grid) - offset(args, first), layout ? 4u : 16u);
    EXPECT_FLOAT_EQ(value<float>(args, "alpha"), 2.5f);
    EXPECT_FLOAT_EQ(value<float>(args, "beta"), -1.25f);
    if(outer < 3)
        EXPECT_LT(offset(args, "beta"), offset(args, first));
    else
        EXPECT_EQ(offset(args, "alpha"), offset(args, first) + (layout ? 8u : 24u));

    for(auto name : {"ws", "Flags", "AddressWS", "AddressFlags"})
        EXPECT_FALSE(hasArgument(args, name)) << name;
    for(auto name : {"magicNumberItersPerTile", "magicShiftItersPerTile", "SKItersPerWG", "skTiles"})
        EXPECT_EQ(hasArgument(args, name), layout == 0) << name;
    if(layout == 0)
    {
        EXPECT_EQ(value<uint32_t>(args, "SKItersPerWG"), 0u);
        EXPECT_EQ(value<uint32_t>(args, "skTiles"), 0u);
    }

    // The XML carries the actual packed names, offsets and widths for the
    // independent Python-signature integration comparison.
    std::ostringstream packed;
    for(auto name : {"gemm_count", "internalArgs", "internalArgs1", "internalArgs2",
                     "numWorkGroups", "size_0", "size_1", "size_2", "size_3",
                     "a", "b", "c", "d", "strideA1", "strideA2", "strideB1", "strideB2",
                     "strideC1", "strideC2", "strideD1", "strideD2", "alpha", "beta",
                     "itersPerTile", "magicNumberItersPerTile", "magicShiftItersPerTile",
                     "SKItersPerWG", "skGrid", "skTiles", "ItersPerTile", "PersistentGrid",
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

INSTANTIATE_TEST_SUITE_P(OuterAndPersistentVersions, PersistentArgumentLayoutTest,
                        ::testing::Values(std::make_tuple(0, 0), std::make_tuple(1, 0),
                                          std::make_tuple(2, 0), std::make_tuple(3, 0),
                                          std::make_tuple(3, 1)));

TEST(PersistentArgumentLayout, NativePayloadChangesOnlySchedulingSlots)
{
    for(int outer : {3})
    {
        SCOPED_TRACE(outer);
        ContractionSolution legacy, native;
        configurePersistentSolution(legacy, outer, 0);
        configurePersistentSolution(native, outer, 1);
        auto problem = persistentProblem();
        auto device = persistentDevice();
        auto oldCall = legacy.generateSingleCall<true>(
            problem, persistentInputs(), device, legacy.resolvePersistentSettings(problem, device), GSUSettings{});
        auto newCall = native.generateSingleCall<true>(
            problem, persistentInputs(), device, native.resolvePersistentSettings(problem, device), GSUSettings{});
        auto const& oldArgs = oldCall.args;
        auto const& newArgs = newCall.args;
        auto oldStart = offset(oldArgs, "itersPerTile");
        auto newStart = offset(newArgs, "ItersPerTile");
        EXPECT_EQ(oldStart, newStart);
        EXPECT_EQ(oldArgs.size(), newArgs.size() + 16u);
        EXPECT_EQ(bytes(oldArgs, 0, oldStart), bytes(newArgs, 0, newStart));
        EXPECT_EQ(bytes(oldArgs, oldStart + 24, oldArgs.size()),
                  bytes(newArgs, newStart + 8, newArgs.size()));
    }
}

TEST(PersistentArgumentLayout, FullTileIterationsAndCoverageIncludeBatchesAndZeroK)
{
    for(auto [m, n, k, batches, grid] : std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>>{
            {1, 1, 0, 1, 64}, {128, 128, 1, 3, 3}, {129, 257, 64, 2, 5},
            {257, 385, 65, 3, 7}, {1024, 1024, 129, 4, 16}})
    {
        for(float alpha : {0.0f, 2.0f})
        {
            SCOPED_TRACE(::testing::Message() << m << ',' << n << ',' << k << ',' << batches << ',' << grid << ',' << alpha);
            ContractionSolution solution;
            configurePersistentSolution(solution, 3, 1);
            auto problem = persistentProblem(m, n, k, batches);
            auto device = persistentDevice(grid);
            auto launch = solution.resolvePersistentSettings(problem, device);
            auto invocation = solution.generateSingleCall<true>(
                problem, persistentInputs(alpha), device, launch, GSUSettings{});
            size_t tiles = ((m + 127) / 128) * ((n + 127) / 128) * batches;
            EXPECT_EQ(launch.totalTiles, tiles);
            EXPECT_EQ(value<uint32_t>(invocation.args, "ItersPerTile"), std::max(size_t{1}, (k + 63) / 64));
            EXPECT_EQ(value<uint32_t>(invocation.args, "PersistentGrid"), grid);
            EXPECT_EQ(invocation.numWorkGroups.x, grid);
            EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);
            EXPECT_FLOAT_EQ(value<float>(invocation.args, "alpha"), alpha);
        }
    }
}

TEST(PersistentArgumentLayout, RejectsUnknownAndIncompatibleLayoutsBeforePacking)
{
    auto problem = persistentProblem();
    auto device = persistentDevice();
    for(int version : {-1, 2, 99})
    {
        ContractionSolution solution;
        configurePersistentSolution(solution, 3, version);
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
    for(auto strategy : {TileProcessingStrategy::None, TileProcessingStrategy::StreamK})
    {
        ContractionSolution solution;
        configurePersistentSolution(solution, 3, 1);
        solution.sizeMapping.tileProcessingStrategy = strategy;
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
    for(int outer : {0, 1, 2})
    {
        ContractionSolution solution;
        configurePersistentSolution(solution, outer, 1);
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
}

TEST(PersistentArgumentLayout, CustomDescriptorPacksNativePayloadAndRejectsLegacyClaims)
{
    ContractionSolution solution;
    configurePersistentSolution(solution, 3, 1);
    solution.customKernel.name = "prebuilt_native_data_parallel";
    solution.kernelName = solution.customKernel.name;
    solution.customKernel.macrotile = TensileLite::dim3(128, 128, 64);
    solution.customKernel.threads = TensileLite::dim3(256, 1, 1);
    solution.customKernel.grid = {CustomGridSize::PersistentGrid, CustomGridSize::One, CustomGridSize::One};
    solution.customKernel.args = {
        {CustomArgType::uint32, CustomArgSemantic::ItersPerTile},
        {CustomArgType::uint32, CustomArgSemantic::PersistentGrid},
    };
    auto problem = persistentProblem();
    auto device = persistentDevice();
    auto launch = solution.resolvePersistentSettings(problem, device);
    auto invocation = solution.generateCustomCall<true>(problem, persistentInputs(), device, launch);
    EXPECT_EQ(invocation.kernelName, "prebuilt_native_data_parallel");
    EXPECT_EQ(invocation.numWorkGroups.x, 7u);
    ASSERT_EQ(invocation.args.size(), 8u);
    EXPECT_EQ(value<uint32_t>(invocation.args, "ItersPerTile"), 3u);
    EXPECT_EQ(value<uint32_t>(invocation.args, "PersistentGrid"), 7u);

    for(auto semantic : {CustomArgSemantic::AddressSynchronizer,
                         CustomArgSemantic::Synchronizer, CustomArgSemantic::GSUSync})
    {
        solution.customKernel.args.push_back({CustomArgType::address, semantic});
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
        solution.customKernel.args.pop_back();
    }

    solution.customKernel.args[1].type = CustomArgType::uint64;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.customKernel.args[1] = {CustomArgType::uint32, CustomArgSemantic::SKGrid};
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.customKernel.args[1] = {CustomArgType::uint32, CustomArgSemantic::PersistentGrid};
    solution.internalArgsSupport.persistentLoopArgsVersion = 0;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
}

TEST(PersistentArgumentLayout, NativeCustomDescriptorMatchesCompleteNormalBuffer)
{
    for(bool initialStrides : {false, true})
    for(bool pointerArray : {false, true})
    for(bool useBeta : {false, true})
    for(size_t k : {size_t{0}, size_t{129}})
    for(float alpha : {0.0f, 2.5f})
    for(auto cluster : {TensileLite::dim3(1, 1, 1), TensileLite::dim3(2, 1, 1),
                        TensileLite::dim3(2, 2, 1)})
    {
        SCOPED_TRACE(::testing::Message() << "initialStrides=" << initialStrides
                     << ", pointerArray=" << pointerArray << ", useBeta=" << useBeta
                     << ", K=" << k << ", alpha=" << alpha
                     << ", cluster=" << cluster.x << 'x' << cluster.y);
        ContractionSolution solution;
        configurePersistentSolution(solution, 3, 1);
        solution.sizeMapping.clusterDim = cluster;
        solution.problemType.useInitialStridesAB = initialStrides;
        solution.problemType.useInitialStridesCD = initialStrides;
        solution.problemType.stridedBatched = !pointerArray;
        solution.problemType.useBeta = useBeta;
        solution.problemType.useBias = 1;
        solution.problemType.useGateResidual = true;
        solution.problemType.gateResidualDataTypeWhiteList = {rocisa::DataType::Float};
        solution.customKernel.name = solution.kernelName;
        solution.customKernel.macrotile = TensileLite::dim3(128, 128, 64);
        solution.customKernel.threads = TensileLite::dim3(256, 1, 1);
        solution.customKernel.grid = {CustomGridSize::PersistentGrid, CustomGridSize::One, CustomGridSize::One};
        auto append = [&](CustomArgType type, CustomArgSemantic semantic) {
            solution.customKernel.args.push_back({type, semantic});
        };
        auto scalar = [&](CustomArgSemantic semantic) { append(CustomArgType::uint32, semantic); };
        auto pointer = [&](CustomArgSemantic semantic) { append(CustomArgType::address, semantic); };
        auto strides = [&](CustomArgSemantic first) {
            for(int i = 0; i < (initialStrides ? 3 : 2); ++i)
                scalar(static_cast<CustomArgSemantic>(static_cast<int>(first) + i));
        };
        for(auto semantic : {CustomArgSemantic::GemmInfo, CustomArgSemantic::InternalArgs,
                             CustomArgSemantic::InternalArgs1, CustomArgSemantic::NumWorkGroups,
                             CustomArgSemantic::SizeFree0, CustomArgSemantic::SizeFree1,
                             CustomArgSemantic::SizeFree2, CustomArgSemantic::SizeSum})
            scalar(semantic);
        pointer(CustomArgSemantic::AddressA);
        pointer(CustomArgSemantic::AddressB);
        strides(CustomArgSemantic::StrideA0);
        strides(CustomArgSemantic::StrideB0);
        scalar(CustomArgSemantic::ItersPerTile);
        scalar(CustomArgSemantic::PersistentGrid);
        scalar(CustomArgSemantic::Alpha);
        scalar(CustomArgSemantic::Beta);
        pointer(CustomArgSemantic::AddressD);
        pointer(CustomArgSemantic::AddressC);
        strides(CustomArgSemantic::StrideD0);
        strides(CustomArgSemantic::StrideC0);
        pointer(CustomArgSemantic::AddressBias);
        scalar(CustomArgSemantic::BiasType);
        scalar(CustomArgSemantic::StrideBias);
        pointer(CustomArgSemantic::AddressGateResidual);
        scalar(CustomArgSemantic::GateResidualType);
        strides(CustomArgSemantic::StrideGate0);
        for(auto semantic : {CustomArgSemantic::BatchOffsetD, CustomArgSemantic::BatchOffsetC,
                             CustomArgSemantic::BatchOffsetA, CustomArgSemantic::BatchOffsetB})
            append(CustomArgType::uint64, semantic);

        auto problem = persistentProblem(257, 385, k, 3);
        problem.setStridedBatched(!pointerArray);
        if(pointerArray)
            problem.setBatchMode(ContractionProblemGemm::BATCHMODE::POINTER_ARRAY);
        problem.setUseBias(1);
        problem.setBias(rocisa::DataType::Float, problem.d().sizes()[0], 521);
        problem.setUseGateResidual(true);
        problem.setGateResidual(rocisa::DataType::Float, problem.d().sizes(), {1, 521, 262144});
        auto device = persistentDevice();
        auto inputs = persistentInputs(alpha);
        float sentinels[6]{};
        void const* batchA[] = {&sentinels[0]};
        void const* batchB[] = {&sentinels[1]};
        void const* batchC[] = {&sentinels[2]};
        void* batchD[] = {&sentinels[3]};
        void const* batchBias[] = {&sentinels[4]};
        void const* batchGate[] = {&sentinels[5]};
        inputs.a = &sentinels[0];
        inputs.b = &sentinels[1];
        inputs.c = &sentinels[2];
        inputs.d = &sentinels[3];
        inputs.bias = &sentinels[4];
        inputs.gateResidual = &sentinels[5];
        inputs.batchA = batchA;
        inputs.batchB = batchB;
        inputs.batchC = batchC;
        inputs.batchD = batchD;
        inputs.batchBias = batchBias;
        inputs.batchGateResidual = batchGate;
        inputs.batchOffsetD = 0x100000001LL;
        inputs.batchOffsetC = 0x200000003LL;
        inputs.batchOffsetA = 0x300000005LL;
        inputs.batchOffsetB = 0x400000007LL;
        auto launch = solution.resolvePersistentSettings(problem, device);
        solution.customKernel.generated = true;
        auto normal = solution.generateSingleCall<true>(problem, inputs, device, launch, GSUSettings{});
        solution.customKernel.generated = false;
        auto custom = solution.generateCustomCall<true>(problem, inputs, device, launch);
        ASSERT_EQ(custom.args.size(), normal.args.size());
        EXPECT_EQ(bytes(custom.args, 0, custom.args.size()), bytes(normal.args, 0, normal.args.size()));
        auto dimensions = [](auto const& value) {
            return std::make_tuple(value.x, value.y, value.z);
        };
        EXPECT_EQ(dimensions(custom.numWorkGroups), dimensions(normal.numWorkGroups));
        EXPECT_EQ(dimensions(custom.workGroupSize), dimensions(normal.workGroupSize));
        EXPECT_EQ(dimensions(custom.numWorkItems), dimensions(normal.numWorkItems));
        EXPECT_EQ(dimensions(custom.clusterDim), dimensions(normal.clusterDim));
        EXPECT_EQ(value<int64_t>(custom.args, "batchOffsetA"), inputs.batchOffsetA);
        EXPECT_EQ(value<void const*>(custom.args, "AddressBias"),
                  pointerArray ? static_cast<void const*>(batchBias) : inputs.bias);
        EXPECT_EQ(value<uint32_t>(custom.args, "StrideA0"), initialStrides ? 1u : problem.a().strides()[1]);
        EXPECT_FLOAT_EQ(value<float>(custom.args, "beta"), useBeta ? -1.25f : 0.0f);
    }
}
