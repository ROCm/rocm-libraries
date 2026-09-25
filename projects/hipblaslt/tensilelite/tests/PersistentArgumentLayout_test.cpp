// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstring>
#include <limits>
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

TEST(PersistentArgumentLayout, DataParallelV1PayloadChangesOnlySchedulingSlots)
{
    for(int outer : {3})
    {
        SCOPED_TRACE(outer);
        ContractionSolution legacy, dataParallelV1;
        configurePersistentSolution(legacy, outer, 0);
        configurePersistentSolution(dataParallelV1, outer, 1);
        auto problem = persistentProblem();
        auto device = persistentDevice();
        auto oldCall = legacy.generateSingleCall<true>(
            problem, persistentInputs(), device, legacy.resolvePersistentSettings(problem, device), GSUSettings{});
        auto newCall = dataParallelV1.generateSingleCall<true>(
            problem, persistentInputs(), device, dataParallelV1.resolvePersistentSettings(problem, device), GSUSettings{});
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

TEST(PersistentArgumentLayout, CustomDescriptorPacksDataParallelPayloadAndRejectsLegacyClaims)
{
    ContractionSolution solution;
    configurePersistentSolution(solution, 3, 1);
    solution.customKernel.name = "prebuilt_data_parallel_v1";
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
    EXPECT_EQ(invocation.kernelName, "prebuilt_data_parallel_v1");
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

TEST(PersistentArgumentLayout, DataParallelCustomDescriptorMatchesCompleteNormalBuffer)
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

namespace
{
    ContractionProblemGemm clusterProblem(size_t m = 257, size_t n = 385,
                                          size_t k = 129, size_t batches = 3)
    {
        auto problem = persistentProblem(m, n, k, batches);
        // A problem built by GEMM starts with a zero workspace budget. Resolve
        // the ideal split first; fallback tests then supply an explicit budget.
        problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
        return problem;
    }

    void configureClusterStreamK(ContractionSolution& solution, uint32_t cs, uint32_t cn)
    {
        configurePersistentSolution(solution, 3, 2);
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::StreamK;
        solution.sizeMapping.streamKClusterMulticast = true;
        solution.sizeMapping.clusterDim = {cs, cn, 1};
        solution.sizeMapping.workspaceSizePerElemC = 4;
    }

    ContractionInputs clusterInputs(float alpha = 2.5f)
    {
        auto inputs = persistentInputs(alpha);
        // Host argument packing never dereferences the addresses.
        inputs.ws = reinterpret_cast<void*>(uintptr_t{0x10000});
        inputs.Synchronizer = reinterpret_cast<void*>(uintptr_t{0x20000});
        return inputs;
    }
}

TEST(PersistentArgumentLayout, ClusterStreamKUsesOneResolvedBlockSchedule)
{
    for(auto [cs, cn] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {4, 1}, {2, 2}, {2, 4}})
    for(auto [m, n, k, batches, physicalBudget] :
        std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>>{
            {1, 1, 0, 1, 256}, {1, 1, 1, 1, 256}, {129, 257, 65, 3, 35},
            {1025, 769, 1025, 2, 32}, {127, 127, 4097, 3, 256},
            {513, 257, 4096, 3, 100000}, {129, 1, 512, 1, 1}})
    {
        SCOPED_TRACE(::testing::Message() << cs << ',' << cn << ':' << m << ',' << n
                     << ',' << k << ',' << batches << ',' << physicalBudget);
        ContractionSolution solution;
        configureClusterStreamK(solution, cs, cn);
        auto problem = clusterProblem(m, n, k, batches);
        auto device = persistentDevice(physicalBudget);
        auto launch = solution.resolvePersistentSettings(problem, device);
        auto const& cluster = launch.clusterSchedule;
        ASSERT_TRUE(cluster.enabled);
        EXPECT_EQ(launch.argsVersion, 2);
        EXPECT_EQ(launch.selectedGrid, physicalBudget);
        EXPECT_EQ(launch.totalTiles, ((m + 127) / 128) * ((n + 127) / 128) * batches);
        EXPECT_EQ(cluster.blocksM, ((m + 127) / 128 + cs - 1) / cs);
        EXPECT_EQ(cluster.blocksN, ((n + 127) / 128 + cn - 1) / cn);
        EXPECT_EQ(cluster.blocks, cluster.blocksM * cluster.blocksN * batches);
        EXPECT_EQ(cluster.physicalGrid, launch.grid * cs * cn);
        EXPECT_LE(cluster.physicalGrid, StreamKFlagElements);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), launch.workspaceBytes);
        EXPECT_EQ(launch.workspaceBytes, cluster.partialSlots * 128 * 128 * sizeof(float));
        EXPECT_EQ(cluster.partialSlots, cluster.flagEntries);
        EXPECT_EQ(launch.reduction, origami::reduction_t::tree);
        for(float alpha : {0.0f, 2.5f})
        {
            auto invocation = solution.generateSingleCall<true>(
                problem, clusterInputs(alpha), device, launch, GSUSettings{});
            EXPECT_EQ(invocation.numWorkGroups.x, cs * launch.grid);
            EXPECT_EQ(invocation.numWorkGroups.y, cn);
            EXPECT_EQ(invocation.numWorkGroups.z, 1u);
            EXPECT_EQ(value<uint32_t>(invocation.args, "numWorkGroups"), cluster.physicalGrid);
            EXPECT_EQ(value<uint32_t>(invocation.args, "itersPerTile"), cluster.itersPerTile);
            EXPECT_EQ(value<uint32_t>(invocation.args, "skGrid"), launch.grid);
            EXPECT_EQ(value<uint32_t>(invocation.args, "skTiles"), cluster.split.skTiles);
            EXPECT_EQ(value<uint32_t>(invocation.args, "SKItersPerWG"), cluster.split.skItersPerWG);
            EXPECT_EQ(value<void*>(invocation.args, "Flags"), clusterInputs().Synchronizer);
            EXPECT_FLOAT_EQ(value<float>(invocation.args, "alpha"), alpha);
        }
        auto decisions = solution.computeStreamKDecisions(problem, device);
        EXPECT_EQ(decisions.skGrid, launch.grid);
        EXPECT_EQ(decisions.requiredWorkspaceBytes, launch.workspaceBytes);
        EXPECT_EQ(decisions.clusterSchedule.physicalGrid, cluster.physicalGrid);
        EXPECT_EQ(decisions.skTiles, cluster.split.skTiles);
        EXPECT_EQ(decisions.partialsPresent, cluster.partialSlots != 0);
        std::ostringstream report;
        solution.printStreamKLaunchSummary(report, problem, decisions);
        for(auto token : {"physicalWGs", "logicalClusters", "spatialBlocks", "skItersPerCluster", "flagEntries"})
            EXPECT_NE(report.str().find(token), std::string::npos);
    }
}

TEST(PersistentArgumentLayout, ClusterStreamKCanSplitOneBlockAcrossClusters)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    auto problem = clusterProblem(129, 257, 4097, 1);
    auto device = persistentDevice(256);
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_GT(launch.grid, launch.clusterSchedule.blocks);
    EXPECT_GT(launch.workspaceBytes, 0u);
    EXPECT_EQ(launch.clusterSchedule.partialSlots, launch.grid * 8);
    const auto expected = streamKStaticSplit(1, 65, launch.grid, device.skFullTiles, false);
    EXPECT_EQ(launch.clusterSchedule.split.skItersPerWG, expected.skItersPerWG);
    EXPECT_EQ(launch.clusterSchedule.split.extraIters, expected.extraIters);
}

TEST(PersistentArgumentLayout, ClusterStreamKWorkspaceFallbackPreservesWholeClusterGeometry)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    auto problem = clusterProblem(513, 769, 4097, 3);
    auto device = persistentDevice(256);
    auto ideal = solution.resolvePersistentSettings(problem, device);
    ASSERT_GT(ideal.workspaceBytes, 0u);
    problem.setWorkspaceSize(ideal.workspaceBytes);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), ideal.workspaceBytes);
    EXPECT_EQ(solution.resolvePersistentSettings(problem, device).grid, ideal.grid);
    problem.setWorkspaceSize(ideal.workspaceBytes - 1);
    auto fallback = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);
    EXPECT_EQ(fallback.workspaceBytes, 0u);
    EXPECT_TRUE(fallback.clusterSchedule.workspaceFallback);
    EXPECT_TRUE(fallback.clusterSchedule.wholeBlocksOnly);
    EXPECT_EQ(fallback.clusterSchedule.split.skTiles, 0u);
    EXPECT_EQ(fallback.clusterSchedule.split.skItersPerWG, 0u);
    EXPECT_EQ(fallback.clusterSchedule.physicalGrid, fallback.grid * 8);
    EXPECT_LE(fallback.grid, fallback.clusterSchedule.blocks);
    EXPECT_EQ(fallback.reduction, origami::reduction_t::tree);
    auto inputs = clusterInputs();
    inputs.ws = nullptr;
    EXPECT_NO_THROW(solution.generateSingleCall<true>(problem, inputs, device, fallback, GSUSettings{}));
    inputs.Synchronizer = nullptr;
    EXPECT_THROW(solution.generateSingleCall<true>(problem, inputs, device, fallback, GSUSettings{}), std::runtime_error);
}

TEST(PersistentArgumentLayout, ClusterStreamKBoundsAndUniformOrderAreExplicit)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 1);
    auto device = persistentDevice(64);
    auto problem = clusterProblem(1, 1, 64 * 65536, 1);
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_TRUE(launch.clusterSchedule.treeBoundsFallback);
    EXPECT_EQ(launch.clusterSchedule.split.skTiles, 0u);
    EXPECT_EQ(launch.workspaceBytes, 0u);
    solution.sizeMapping.depthU = 1;
    problem = clusterProblem(1, 1, 1 << 24, 1);
    EXPECT_TRUE(solution.resolvePersistentSettings(problem, device).clusterSchedule.treeBoundsFallback);
    problem = clusterProblem(size_t{128} << 16, size_t{128} << 16, 1, 1);
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::overflow_error);
    problem = clusterProblem(1, 1, 65, 1);
    solution.sizeMapping.workspaceSizePerElemC = std::numeric_limits<size_t>::max();
    EXPECT_THROW(solution.requiredWorkspaceSize(problem, device), std::overflow_error);
    solution.sizeMapping.workspaceSizePerElemC = sizeof(float);
    problem.setParams().setUniformSummationOrder(true);
    EXPECT_FALSE(solution.uniformSummationOrderSupported(problem, device));
}

TEST(PersistentArgumentLayout, ClusterStreamKRequiresVersionTwoAndSupportedCapabilities)
{
    auto problem = clusterProblem();
    auto device = persistentDevice();
    for(int version : {0, 1})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, 2, 1);
        solution.internalArgsSupport.persistentLoopArgsVersion = version;
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
    for(int outer : {0, 1, 2})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, 2, 1);
        solution.internalArgsSupport.version = outer;
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
    for(auto assignment : {WorkAssignment::DynamicWorkQueue, WorkAssignment::Hybrid})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, 2, 1);
        solution.sizeMapping.workAssignment = assignment;
        EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    }
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 1);
    solution.sizeMapping.prefetchAcrossPersistent = 1;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.sizeMapping.prefetchAcrossPersistent = 0;
    solution.sizeMapping.streamKAtomic = 1;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.sizeMapping.streamKAtomic = 0;
    solution.customKernel.name = "external_abi2";
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.customKernel.name.clear();
    solution.problemType.outputAmaxD = true;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.problemType.outputAmaxD = false;
    solution.problemType.useBias = 1;
    EXPECT_NO_THROW(solution.resolvePersistentSettings(problem, device));
    solution.problemType.useGradient = true;
    EXPECT_THROW(solution.resolvePersistentSettings(problem, device), std::runtime_error);
    solution.problemType.useBias = 0;
    EXPECT_NO_THROW(solution.resolvePersistentSettings(problem, device));
}

TEST(PersistentArgumentLayout, ClusterStreamKFlagCapacityAndLargeWorkspace)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    solution.sizeMapping.macroTile = TensileLite::dim3(1024, 1024, 1);
    auto problem = clusterProblem(2048, 4096, 2049 * 64, 1);
    const size_t tileBytes = size_t{1024} * 1024 * sizeof(float);
    for(size_t budget : {size_t{StreamKFlagElements - 1}, size_t{StreamKFlagElements},
                         size_t{StreamKFlagElements + 1}, size_t{2 * StreamKFlagElements}})
    {
        auto device = persistentDevice(budget);
        // Keep residency above the flag cap so this independently exercises
        // physical-slot capacity and rounding the selected budget to clusters.
        device.computeUnitCount = 2 * StreamKFlagElements;
        const auto launch = solution.resolvePersistentSettings(problem, device);
        const size_t slots = std::min(budget, size_t{StreamKFlagElements}) / 8 * 8;
        ASSERT_EQ(launch.clusterSchedule.flagEntries, slots);
        EXPECT_EQ(launch.clusterSchedule.partialSlots, slots);
        EXPECT_EQ(launch.clusterSchedule.physicalGrid, slots);
        EXPECT_EQ(launch.selectedGrid, budget);
        EXPECT_EQ(launch.grid, slots / 8);
        ASSERT_GT(launch.workspaceBytes, size_t{1} << 32);
        EXPECT_EQ(launch.workspaceBytes, slots * tileBytes);
        problem.setWorkspaceSize(launch.workspaceBytes);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), slots * tileBytes);
        problem.setWorkspaceSize(launch.workspaceBytes - 1);
        const auto fallback = solution.resolvePersistentSettings(problem, device);
        EXPECT_TRUE(fallback.clusterSchedule.workspaceFallback);
        EXPECT_EQ(fallback.workspaceBytes, 0u);
        EXPECT_EQ(fallback.clusterSchedule.flagEntries, 0u);
        EXPECT_EQ(fallback.clusterSchedule.split.skTiles, 0u);
        EXPECT_EQ(fallback.clusterSchedule.physicalGrid, 8u);
        problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    }
}

TEST(PersistentArgumentLayout, ClusterStreamKGeometryFollowsOutputIndexOrder)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    solution.sizeMapping.macroTile = TensileLite::dim3(64, 128, 1);
    auto problem = ContractionProblemGemm::FromIndexSizes(
        "Contraction_l_Ajlk_Blik_Cijk_Dijk", {129, 513, 3, 4097},
        rocisa::DataType::Float, {}, rocisa::DataType::Float, {},
        rocisa::DataType::Float, {}, rocisa::DataType::Float, {}, 1.0);
    ASSERT_TRUE(problem.transposeC01());
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    const auto launch = solution.resolvePersistentSettings(problem, persistentDevice(64));
    // Output indices i,j map to macroTile.x,y before spatial blocking.
    EXPECT_EQ(launch.totalTiles, 3u * 5u * 3u);
    EXPECT_EQ(launch.clusterSchedule.blocksM, 2u);
    EXPECT_EQ(launch.clusterSchedule.blocksN, 2u);
    EXPECT_EQ(launch.clusterSchedule.blocks, 12u);
}

TEST(PersistentArgumentLayout, SharedClusterGeometryPreservesDataParallelGridAndPap)
{
    // 3 x 4 real tiles in each of three batches. The expected grids are
    // physical workgroups, including the boundary block's phantom peers.
    for(auto [cs, cn, budget, grid, papGrid] :
        std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>>{
            {2, 1, 7, 6, 48}, {4, 1, 7, 4, 48},
            {2, 2, 7, 4, 48}, {2, 4, 7, 8, 48},
            {2, 1, 256, 48, 48}, {2, 4, 256, 48, 48}})
    for(bool pap : {false, true})
    {
        ContractionSolution solution;
        configurePersistentSolution(solution, 3, 1);
        solution.sizeMapping.clusterDim = {cs, cn, 1};
        solution.sizeMapping.prefetchAcrossPersistent = pap;
        auto problem = persistentProblem(257, 385, 129, 3);
        auto device = persistentDevice(budget);
        auto launch = solution.resolvePersistentSettings(problem, device);
        const size_t expectedGrid = pap ? papGrid : grid;
        EXPECT_EQ(launch.selectedGrid, budget);
        EXPECT_EQ(launch.grid, expectedGrid);
        EXPECT_EQ(launch.totalTiles, 36u);
        auto invocation = solution.generateSingleCall<true>(
            problem, persistentInputs(), device, launch, GSUSettings{});
        EXPECT_EQ(invocation.numWorkGroups.x, expectedGrid / cn);
        EXPECT_EQ(invocation.numWorkGroups.y, cn);
        EXPECT_EQ(invocation.numWorkGroups.z, 1u);
        EXPECT_EQ(value<uint32_t>(invocation.args, "PersistentGrid"), expectedGrid);
        // Preserve the legacy DP launch-header convention in this extraction.
        EXPECT_EQ(value<uint32_t>(invocation.args, "numWorkGroups"), expectedGrid / cn);
    }
}
