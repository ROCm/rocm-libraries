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
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

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
        solution.sizeMapping.waveNum = 4;
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
        EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error);
        solution.customKernel.args.pop_back();
    }

    solution.customKernel.args[1].type = CustomArgType::uint64;
    EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error);
    solution.customKernel.args[1] = {CustomArgType::uint32, CustomArgSemantic::SKGrid};
    EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error);
    solution.customKernel.args[1] = {CustomArgType::uint32, CustomArgSemantic::PersistentGrid};
    solution.internalArgsSupport.persistentLoopArgsVersion = 0;
    EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error);
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
        // DP keeps the launch-header count in X workgroups, not physical peers.
        EXPECT_EQ(value<uint32_t>(normal.args, "numWorkGroups"), launch.grid / cluster.y);
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

TEST(PersistentArgumentLayout, ClusterStreamKPacksRaggedBatchedSchedule)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    // Two batches of 2x3 real tiles occupy two eight-peer blocks. The physical
    // budget rounds from 35 to 32: four clusters split 2 * 65 K iterations.
    auto problem = clusterProblem(129, 257, 4097, 2);
    auto device = persistentDevice(35);
    const auto launch = solution.resolvePersistentSettings(problem, device);
    const auto& cluster = launch.clusterSchedule;
    EXPECT_EQ(launch.argsVersion, 2);
    EXPECT_EQ(launch.selectedGrid, 35u);
    EXPECT_EQ(launch.totalTiles, 12u);
    EXPECT_EQ(cluster.blocksM, 1u);
    EXPECT_EQ(cluster.blocksN, 1u);
    EXPECT_EQ(cluster.blocks, 2u);
    EXPECT_EQ(launch.grid, 4u);
    EXPECT_EQ(cluster.physicalGrid, 32u);
    EXPECT_EQ(cluster.flagEntries, 32u);
    EXPECT_EQ(cluster.partialSlots, 32u);
    EXPECT_EQ(cluster.split.extraIters, 2u);
    EXPECT_EQ(launch.workspaceBytes, 32u * 128 * 128 * sizeof(float));
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), launch.workspaceBytes);
    EXPECT_EQ(launch.reduction, origami::reduction_t::tree);
    const auto inputs = clusterInputs();
    const auto call = solution.generateSingleCall<true>(problem, inputs, device, launch, GSUSettings{});
    EXPECT_EQ(call.numWorkGroups.x, 8u);
    EXPECT_EQ(call.numWorkGroups.y, 4u);
    EXPECT_EQ(call.numWorkGroups.z, 1u);
    EXPECT_EQ(value<uint32_t>(call.args, "numWorkGroups"), 32u);
    EXPECT_EQ(value<uint32_t>(call.args, "itersPerTile"), 65u);
    EXPECT_EQ(value<uint32_t>(call.args, "skGrid"), 4u);
    EXPECT_EQ(value<uint32_t>(call.args, "skTiles"), 2u);
    EXPECT_EQ(value<uint32_t>(call.args, "SKItersPerWG"), 32u);
    EXPECT_EQ(value<void*>(call.args, "Flags"), inputs.Synchronizer);
    EXPECT_FLOAT_EQ(value<float>(call.args, "alpha"), 2.5f);
    const auto decisions = solution.computeStreamKDecisions(problem, device);
    EXPECT_EQ(decisions.skGrid, 4u);
    EXPECT_EQ(decisions.skTiles, 2u);
    EXPECT_TRUE(decisions.partialsPresent);
    EXPECT_EQ(decisions.requiredWorkspaceBytes, launch.workspaceBytes);
}

TEST(PersistentArgumentLayout, ClusterStreamKRoundsBudgetToCompleteClusters)
{
    for(auto [cs, cn] : {std::pair<uint32_t, uint32_t>{2, 4}, {1, 8}})
    {
        SCOPED_TRACE(::testing::Message() << cs << "," << cn);
        ContractionSolution solution;
        configureClusterStreamK(solution, cs, cn);
        const auto problem = clusterProblem(cs == 1 ? 128 : 129, 257, 4097, 1);
        // One block has 65 iterations. A sub-cluster budget still launches every
        // peer; larger budgets split it into 2 or 4 workers with one extra iteration.
        for(auto [budget, grid, iters, extra] :
            std::vector<std::tuple<size_t, size_t, uint32_t, uint32_t>>{
                {1, 1, 0, 0}, {15, 1, 0, 0}, {16, 2, 32, 1}, {35, 4, 16, 1}})
        {
            SCOPED_TRACE(budget);
            const auto launch = solution.resolvePersistentSettings(problem, persistentDevice(budget));
            EXPECT_EQ(launch.selectedGrid, budget);
            EXPECT_EQ(launch.grid, grid);
            EXPECT_EQ(launch.clusterSchedule.physicalGrid, grid * 8);
            EXPECT_EQ(launch.clusterSchedule.split.skItersPerWG, iters);
            EXPECT_EQ(launch.clusterSchedule.split.extraIters, extra);
            EXPECT_EQ(launch.workspaceBytes, grid == 1 ? 0 : grid * 8 * 128 * 128 * sizeof(float));
        }
    }
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

TEST(PersistentArgumentLayout, ClusterStreamKRejectsOverflowAndBoundsTreeArithmetic)
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
}

TEST(PersistentArgumentLayout, ClusterStreamKVersionZeroKeepsPrebuiltScheduling)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 1);
    solution.internalArgsSupport.persistentLoopArgsVersion = 0;
    EXPECT_FALSE(solution.usesStreamKSpatialCluster());
    EXPECT_NO_THROW(solution.validatePersistentLoopArgs());

    auto problem = clusterProblem();
    auto device = persistentDevice(8);
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(launch.argsVersion, 0);
    EXPECT_FALSE(launch.clusterSchedule.enabled);
    EXPECT_EQ(launch.totalTiles, 36u);
    EXPECT_EQ(launch.grid, 8u);
    auto invocation = solution.generateSingleCall<true>(
        problem, clusterInputs(), device, launch, GSUSettings{});
    EXPECT_EQ(invocation.numWorkGroups.x, 8u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    EXPECT_EQ(invocation.clusterDim.x, 2u);
    EXPECT_EQ(invocation.clusterDim.y, 1u);
    EXPECT_EQ(value<uint32_t>(invocation.args, "skGrid"), 8u);
    const auto expected = streamKStaticSplit(36, 3, 8, device.skFullTiles, false);
    EXPECT_EQ(value<uint32_t>(invocation.args, "skTiles"), expected.skTiles);
    EXPECT_EQ(value<uint32_t>(invocation.args, "SKItersPerWG"), expected.skItersPerWG);

    solution.internalArgsSupport.persistentLoopArgsVersion = 2;
    EXPECT_TRUE(solution.usesStreamKSpatialCluster());
    auto spatial = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(spatial.argsVersion, 2);
    EXPECT_TRUE(spatial.clusterSchedule.enabled);
    EXPECT_EQ(spatial.clusterSchedule.blocks, 24u);
    EXPECT_EQ(spatial.grid, 4u);
}

TEST(PersistentArgumentLayout, ClusterStreamKVersionTwoRequiresSupportedCapabilities)
{
    auto problem = clusterProblem();
    auto device = persistentDevice();
    for(int version : {-1, 1, 3})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, 2, 1);
        solution.internalArgsSupport.persistentLoopArgsVersion = version;
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
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), slots * tileBytes);
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

TEST(PersistentArgumentLayout, ClusterUniformOrderKeepsIdenticalSplitSignatures)
{
    for(auto shape : {TensileLite::dim3(2, 1, 1), TensileLite::dim3(4, 2, 1),
                      TensileLite::dim3(2, 8, 1)})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, shape.x, shape.y);
        // Ragged spatial blocks and two batches still have identical K folds.
        auto problem = clusterProblem(129, 129, 65 * 64, 2);
        auto device = persistentDevice(256);
        device.computeUnitCount = 256;
        problem.setParams().setUniformSummationOrder(true);
        const auto launch = solution.resolvePersistentSettings(problem, device);
        const auto& cluster = launch.clusterSchedule;
        ASSERT_GT(cluster.partialSlots, 0u);
        ASSERT_GT(launch.grid, cluster.blocks);
        EXPECT_EQ(cluster.split.extraIters, 0u);
        EXPECT_EQ(cluster.split.skTiles, cluster.blocks);
        EXPECT_TRUE(streamKStaticSplitRowUniform(cluster.split, cluster.blocks,
            cluster.itersPerTile, launch.grid, false, true));
        EXPECT_TRUE(solution.uniformSummationOrderSupported(problem, device));
        const auto call = solution.generateSingleCall<true>(
            problem, clusterInputs(), device, launch, GSUSettings{});
        EXPECT_EQ(value<uint32_t>(call.args, "magicShiftItersPerTile") & 0x20000000u, 0u);
        // Capability checks still refuse coordinate-dependent partial storage.
        solution.problemType.useInitialStridesCD = true;
        EXPECT_FALSE(solution.uniformSummationOrderSupported(problem, device));
    }
    const auto irregular = streamKStaticSplit(3, 65, 7, 1, false);
    EXPECT_FALSE(streamKStaticSplitRowUniform(irregular, 3, 65, 7, false, true));
}

TEST(PersistentArgumentLayout, ClusterStreamKWholeTileGridDoesNotCreateResidentTreeSplits)
{
    // Reproduce the 4096-square regression: 4096 whole tiles were clamped to
    // 1536 resident WGs, creating 24 MiB of tree partials in the 2x1 case.
    for(auto [cs, cn] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {2, 2}, {4, 2}})
    {
        ContractionSolution solution;
        configureClusterStreamK(solution, cs, cn);
        solution.sizeMapping.macroTile = {64, 64, 1};
        solution.sizeMapping.workGroupSize = {128, 1, 1};
        solution.sizeMapping.CUOccupancy = 6;
        auto device = persistentDevice(4096);
        device.computeUnitCount = 256;
        auto problem = clusterProblem(4096, 4096, 1024, 1);
        problem.setWorkspaceSize(0);

        const auto launch = solution.resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.selectedGrid, 4096u);
        EXPECT_EQ(launch.clusterSchedule.physicalGrid, 4096u);
        EXPECT_EQ(launch.grid, 4096u / (cs * cn));
        EXPECT_TRUE(launch.clusterSchedule.wholeBlocksOnly);
        EXPECT_FALSE(launch.clusterSchedule.workspaceFallback);
        EXPECT_EQ(launch.clusterSchedule.split.skTiles, 0u);
        EXPECT_EQ(launch.clusterSchedule.flagEntries, 0u);
        EXPECT_EQ(launch.workspaceBytes, 0u);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);

        auto inputs = clusterInputs();
        inputs.ws = nullptr;
        const auto call = solution.generateSingleCall<true>(
            problem, inputs, device, launch, GSUSettings{});
        EXPECT_EQ(call.numWorkGroups.x * call.numWorkGroups.y, 4096u);

        // A genuine K-split request must still respect the cooperative limit.
        problem = clusterProblem(128, 128, 8192, 1);
        const auto split = solution.resolvePersistentSettings(problem, device);
        EXPECT_LE(split.clusterSchedule.physicalGrid, 1536u);
        EXPECT_GT(split.clusterSchedule.partialSlots, 0u);
    }
}

TEST(PersistentArgumentLayout, ClusterTreeDoesNotSerializeIndependentBlocks)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 8, 1);
    solution.sizeMapping.macroTile = {64, 64, 1};
    solution.sizeMapping.workGroupSize = {128, 1, 1};
    solution.sizeMapping.CUOccupancy = 6;
    auto device = persistentDevice(0);
    device.computeUnitCount = 8;
    auto problem = clusterProblem(128, 128, 1024, 1);
    const auto launch = solution.resolvePersistentSettings(problem, device);
    ASSERT_EQ(launch.selectedGrid, 8u);
    ASSERT_EQ(launch.totalTiles, 4u);
    ASSERT_EQ(launch.clusterSchedule.blocks, 2u);
    EXPECT_EQ(launch.grid, 2u);
    EXPECT_EQ(launch.clusterSchedule.physicalGrid, 16u);
    EXPECT_TRUE(launch.clusterSchedule.wholeBlocksOnly);
    EXPECT_EQ(launch.workspaceBytes, 0u);

    device.persistentFixedGrid = 8;
    const auto fixed = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(fixed.grid, 1u);
    EXPECT_EQ(fixed.clusterSchedule.physicalGrid, 8u);
}

TEST(PersistentArgumentLayout, ClusterCostAvoidsTreeFixupWithoutRemovingUsefulSplits)
{
    hip::HipAMDGPU device;
    device.processor = AMDGPU::Processor::gfx1250;
    device.computeUnitCount = 256;
    device.persistentDynamicGrid = 6;
    device.persistentFixedGrid = 0;
    device.analyticalHardware = std::make_shared<origami::hardware_t>(
        origami::hardware_t::architecture_t::gfx950, 256, 163840, 262144,
        8, 1.0, 1.0, 1.0, 4000000, 1.2, 1, std::make_tuple(0.0, 0.008, 0.0));
    for(auto [m, n, batch, k, cs, cn, whole] :
        std::vector<std::tuple<size_t, size_t, size_t, size_t, uint32_t, uint32_t, bool>>{
            {257, 129, 8, 4096, 4, 1, true}, {128, 128, 1, 8192, 4, 1, false},
            // At this ragged two-axis boundary, whole blocks measured slower:
            // 66 whole-block iterations versus 41 + 24 estimated split cost.
            {512, 832, 1, 4161, 2, 4, false}})
    {
        SCOPED_TRACE(::testing::Message() << m << ',' << n << ',' << batch << ',' << k << ',' << cs << ',' << cn);
        ContractionSolution solution;
        configureClusterStreamK(solution, cs, cn);
        solution.sizeMapping.macroTile = {64, 64, 1};
        solution.sizeMapping.workGroupSize = {128, 1, 1};
        solution.sizeMapping.CUOccupancy = 6;
        solution.internalArgsSupport.perTileExtraIters = true;
        auto problem = clusterProblem(m, n, k, batch);
        const auto launch = solution.resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.clusterSchedule.wholeBlocksOnly, whole);
        EXPECT_EQ(launch.workspaceBytes == 0, whole);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), launch.workspaceBytes);
        if(whole)
        {
            EXPECT_EQ(launch.clusterSchedule.fallbackReason, "whole-block-cost");
            EXPECT_EQ(launch.grid, launch.clusterSchedule.blocks);
            EXPECT_EQ(launch.clusterSchedule.flagEntries, 0u);
            EXPECT_EQ(launch.clusterSchedule.split.skTiles, 0u);
        }
        // Explicit fixed grids bypass the performance policy.
        device.persistentFixedGrid = launch.selectedGrid;
        const auto fixed = solution.resolvePersistentSettings(problem, device);
        EXPECT_TRUE(fixed.clusterSchedule.fixedGridUsed);
        EXPECT_NE(fixed.clusterSchedule.fallbackReason, "whole-block-cost");
        device.persistentFixedGrid = 0;
        // A CU cap bounds the new whole-block candidate as well as splits.
        problem.setParams().setSmCountTarget(32);
        const auto capped = solution.resolvePersistentSettings(problem, device);
        EXPECT_LE(capped.clusterSchedule.physicalGrid, 32u * 6u);
    }
}

TEST(PersistentArgumentLayout, ClusterFitCountsRealPeersInOutputOrder)
{
    ContractionSolution solution;
    configureClusterStreamK(solution, 4, 2);
    auto problem = clusterProblem(129, 257, 1024, 3);
    // 2x3 tiles occupy one 4x2 block in M and two in N, per batch.
    EXPECT_DOUBLE_EQ(solution.streamKClusterUtilization(problem), 6.0 / 16.0);
    problem = ContractionProblemGemm::FromIndexSizes(
        "Contraction_l_Ajlk_Blik_Cijk_Dijk", {257, 129, 3, 1024},
        rocisa::DataType::Float, {}, rocisa::DataType::Float, {},
        rocisa::DataType::Float, {}, rocisa::DataType::Float, {}, 1.0);
    ASSERT_TRUE(problem.transposeC01());
    // Transposing output gives 3x2 tiles and a single complete spatial block.
    EXPECT_DOUBLE_EQ(solution.streamKClusterUtilization(problem), 6.0 / 8.0);
    solution.internalArgsSupport.persistentLoopArgsVersion = 0;
    EXPECT_DOUBLE_EQ(solution.streamKClusterUtilization(problem), 1.0);
}

TEST(PersistentArgumentLayout, ClusterParallelEligibilityAndPackedOutputContract)
{
    using arch_t = origami::hardware_t::architecture_t;
    hip::HipAMDGPU device;
    device.processor = AMDGPU::Processor::gfx950;
    device.computeUnitCount = 256;
    device.persistentDynamicGrid = 6;
    device.persistentFixedGrid = 128;
    device.analyticalHardware = std::make_shared<origami::hardware_t>(
        arch_t::gfx950, 256, 163840, 262144, 8, 1.0, 1.0, 1.0, 4000000, 1.2, 1,
        std::make_tuple(0.0, 0.008, 0.0));
    ContractionSolution solution;
    configureClusterStreamK(solution, 2, 4);
    auto problem = clusterProblem(257, 385, 65537, 2);
    ASSERT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::parallel);
    const auto launch = solution.resolvePersistentSettings(problem, device);
    ASSERT_EQ(launch.reduction, origami::reduction_t::parallel);
    auto const& cluster = launch.clusterSchedule;
    ASSERT_EQ(cluster.blocks, 4u);
    EXPECT_EQ(launch.grid, 16u);
    EXPECT_EQ(cluster.physicalGrid, 128u);
    EXPECT_EQ(cluster.parallelSplits, 4u);
    EXPECT_EQ(cluster.flagEntries, 0u);
    EXPECT_EQ(launch.workspaceBytes, 257u * 385 * 2 * sizeof(float) * 4);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), launch.workspaceBytes);
    auto inputs = clusterInputs();
    inputs.Synchronizer = nullptr;
    const auto gemm = solution.generateSingleCall<true>(problem, inputs, device, launch, GSUSettings{});
    EXPECT_EQ(value<void*>(gemm.args, "Flags"), nullptr);
    EXPECT_EQ(value<void*>(gemm.args, "ws_d"), inputs.ws);
    EXPECT_EQ(value<uint32_t>(gemm.args, "skTiles"), 4u);
    EXPECT_EQ(value<uint32_t>(gemm.args, "SKItersPerWG"), 1025u / 4);
    EXPECT_EQ(value<uint32_t>(gemm.args, "skGrid"), 16u);
    const auto conversion = solution.generateOutputConversionCall<true>(problem, inputs, launch, 1, 0);
    EXPECT_EQ(value<uint32_t>(conversion.args, "gsu"), 4u);
    EXPECT_EQ(value<uint32_t>(conversion.args, "strideW1"), 257u);
    EXPECT_EQ(value<uint32_t>(conversion.args, "strideW2"), 257u * 385);
    EXPECT_FLOAT_EQ(value<float>(conversion.args, "alpha"), 2.5f);
    EXPECT_FLOAT_EQ(value<float>(conversion.args, "beta"), -1.25f);
    const auto decisions = solution.computeStreamKDecisions(problem, device);
    EXPECT_EQ(decisions.skSplit, 4u);
    EXPECT_EQ(decisions.requiredWorkspaceBytes, launch.workspaceBytes);

    problem.setWorkspaceSize(launch.workspaceBytes - 1);
    const auto fallback = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(fallback.reduction, origami::reduction_t::tree);
    EXPECT_TRUE(fallback.clusterSchedule.workspaceFallback);
    EXPECT_TRUE(fallback.clusterSchedule.wholeBlocksOnly);
    EXPECT_EQ(fallback.workspaceBytes, 0u);
    EXPECT_EQ(fallback.clusterSchedule.parallelSplits, 1u);
    EXPECT_EQ(fallback.clusterSchedule.physicalGrid, 32u);
    solution.problemType.useGradient = true;
    EXPECT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::tree);
    solution.problemType.useGradient = false;
    solution.problemType.sparse = 1;
    EXPECT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::tree);
    solution.problemType.sparse = 0;
    solution.sizeMapping.useSubtileImpl = true;
    EXPECT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::tree);
    solution.sizeMapping.useSubtileImpl = false;
    solution.customKernel.name = "handwritten";
    EXPECT_EQ(solution.getSKReduction(problem, device), origami::reduction_t::tree);
}

TEST(PersistentArgumentLayout, ClusterParallelPreservesUsefulSplitsAtSpatialEdges)
{
    using arch_t = origami::hardware_t::architecture_t;
    hip::HipAMDGPU device;
    device.processor = AMDGPU::Processor::gfx950;
    device.computeUnitCount = 256;
    device.persistentDynamicGrid = 6;
    device.persistentFixedGrid = 0;
    device.analyticalHardware = std::make_shared<origami::hardware_t>(
        arch_t::gfx950, 256, 163840, 262144, 8, 1.0, 1.0, 1.0, 4000000, 1.2, 1,
        std::make_tuple(0.0, 0.008, 0.0));
    // Two real tiles per axis: no padding, M padding, N padding, then both axes.
    // Padded peers must preserve the predictor's useful K splits.
    for(auto [cs, cn] : std::vector<std::pair<uint32_t, uint32_t>>{
            {2, 1}, {4, 1}, {2, 4}, {4, 4}, {1, 2}, {1, 4}, {1, 8}})
    {
        SCOPED_TRACE(::testing::Message() << cs << ',' << cn);
        ContractionSolution solution;
        configureClusterStreamK(solution, cs, cn);
        solution.sizeMapping.macroTile = {64, 64, 1};
        solution.sizeMapping.matrixInstruction = {16, 16, 32, 1};
        solution.sizeMapping.workGroupSize = {128, 1, 1};
        solution.sizeMapping.CUOccupancy = 6;
        auto problem = clusterProblem(128, 128, 8192, 1);
        const auto launch = solution.resolvePersistentSettings(problem, device);
        ASSERT_EQ(launch.reduction, origami::reduction_t::parallel);
        ASSERT_GT(launch.selectedGrid, launch.totalTiles);
        const auto expectedSplits = launch.selectedGrid / launch.totalTiles;
        const auto& cluster = launch.clusterSchedule;
        EXPECT_EQ(cluster.parallelSplits, expectedSplits);
        EXPECT_EQ(cluster.physicalGrid, cluster.blocks * cs * cn * expectedSplits);
        EXPECT_EQ(cluster.partialSlots, launch.totalTiles * expectedSplits);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), launch.workspaceBytes);
        const auto call = solution.generateSingleCall<true>(
            problem, clusterInputs(), device, launch, GSUSettings{});
        EXPECT_EQ(value<uint32_t>(call.args, "skTiles"), expectedSplits);
        EXPECT_EQ(call.numWorkGroups.x * call.numWorkGroups.y, cluster.physicalGrid);
    }
}
