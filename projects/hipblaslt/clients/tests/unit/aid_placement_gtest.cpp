// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only tests of the die bit: amdgpu 6.19 and newer have the xgmi_physical_id
// rule, and selection of a custom kernel that takes the bit as its AIDPlacementBit
// argument, which only those versions may run.

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/Distance.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/MatchingLibrary.hpp>
#include <Tensile/PropertyMatching.hpp>
#include <Tensile/SwizzleLayout.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/msgpack/Writing.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace
{
    using Gemm     = TensileLite::ContractionProblemGemm;
    using Solution = TensileLite::ContractionSolution;
    using Master   = TensileLite::MasterSolutionLibrary<Gemm, Solution>;
    using Key      = std::array<int64_t, 4>;
    using TensileLite::hip::aidPlacementBit;
    using TensileLite::hip::page24RuleHolds;

    constexpr Key  kKey     = {4096, 4096, 1, 65536};
    constexpr auto kLayoutA = static_cast<int>(TensileLite::SwizzleLayout::P48Page24);
    constexpr auto kLayoutB = static_cast<int>(TensileLite::SwizzleLayout::P48Page24Split);

    const std::vector<std::string> kRejected
        = {"6.16.13", "6.16", "6.18", "6.0", "5.19", "6", "", "amdgpu"};
    const std::vector<std::string> kAccepted = {"6.19.14.31400000",
                                                "6.19",
                                                "6.20",
                                                "6.190",
                                                "7.0",
                                                "7.1",
                                                "7.1.9.31600000",
                                                "7.10",
                                                "8.0"};

    Gemm page24Problem()
    {
        auto problem = Gemm::GEMM_Strides(true,
                                          false,
                                          rocisa::DataType::Float6,
                                          rocisa::DataType::Float6,
                                          rocisa::DataType::BFloat16,
                                          rocisa::DataType::BFloat16,
                                          4096,
                                          4096,
                                          65536,
                                          1,
                                          65536,
                                          65536 * 4096,
                                          65536,
                                          65536 * 4096,
                                          4096,
                                          4096 * 4096,
                                          4096,
                                          4096 * 4096,
                                          0.0);
        problem.setSwizzleTensorA(true);
        problem.setSwizzleTensorB(true);
        problem.setMXScaleFormat(3);
        problem.setSwizzleLayoutA(kLayoutA);
        problem.setSwizzleLayoutB(kLayoutB);
        return problem;
    }

    // One page24 custom kernel at kKey in an Equality table, written to msgpack and loaded
    // back. withAid gives it the kernel's AIDPlacementBit argument.
    std::shared_ptr<Master> page24Library(bool withAid)
    {
        namespace P = TensileLite::Predicates::Contraction;
        namespace C = TensileLite::Contraction;
        using TensileLite::CustomArgSemantic;
        using TensileLite::CustomArgType;

        auto s                        = std::make_shared<Solution>();
        s->index                      = 0;
        s->solutionName               = "ka941";
        s->kernelName                 = s->solutionName;
        s->problemType.swizzleLayoutA = kLayoutA;
        s->problemType.swizzleLayoutB = kLayoutB;
        s->customKernel.name          = s->solutionName;
        s->customKernel.macrotile.x   = 256;
        s->customKernel.macrotile.y   = 256;
        s->customKernel.macrotile.z   = 128;
        s->customKernel.threads.x     = 256;
        s->customKernel.threads.y     = 1;
        s->customKernel.threads.z     = 1;
        s->customKernel.grid.x        = TensileLite::CustomGridSize::TilesXY;
        s->customKernel.uniformSummationOrder = true;
        s->customKernel.args          = {{CustomArgType::address, CustomArgSemantic::AddressB},
                                         {CustomArgType::address, CustomArgSemantic::AddressA},
                                         {CustomArgType::address, CustomArgSemantic::AddressMXScaleB},
                                         {CustomArgType::address, CustomArgSemantic::AddressMXScaleA},
                                         {CustomArgType::address, CustomArgSemantic::AddressD}};
        if(withAid)
            s->customKernel.args.push_back(
                {CustomArgType::int32, CustomArgSemantic::AIDPlacementBit});
        // Selection evaluates the launch settings, which read the tile shape.
        auto& m             = s->sizeMapping;
        m.waveNum           = 4;
        m.workGroupSize     = {256, 1, 1};
        m.threadTile        = {1, 1, 1};
        m.macroTile         = {256, 256, 1};
        m.matrixInstruction = {16, 16, 128, 1};
        m.waveGroup         = {2, 2};
        m.depthU            = 128;
        m.globalSplitU      = 1;
        m.workGroupMapping  = 8;

        auto master          = std::make_shared<Master>();
        master->solutions[0] = s;

        using Table  = TensileLite::Matching::DistanceMatchingTable<
            Key,
            Gemm,
            TensileLite::ContractionProblemMatchingLibrary::Element,
            std::shared_ptr<Solution>,
            TensileLite::Matching::Equality<Key>>;
        auto freeA   = std::make_shared<C::FreeSizeA>();
        auto freeB   = std::make_shared<C::FreeSizeB>();
        auto batch   = std::make_shared<C::BatchSize>();
        auto bound   = std::make_shared<C::BoundSize>();
        freeA->index = freeB->index = batch->index = bound->index = 0;
        auto table        = std::make_shared<Table>();
        table->properties = {freeA, freeB, batch, bound};
        table->table.push_back(
            {kKey, std::make_shared<TensileLite::SingleContractionLibrary>(s), 1.0});
        auto equality   = std::make_shared<TensileLite::ContractionProblemMatchingLibrary>();
        equality->table = table;

        std::vector<std::shared_ptr<TensileLite::Predicates::Predicate<Gemm>>> preds{
            std::make_shared<P::SwizzleTensorA>(true),
            std::make_shared<P::SwizzleTensorB>(true),
            std::make_shared<P::SwizzleLayoutA>(kLayoutA),
            std::make_shared<P::SwizzleLayoutB>(kLayoutB)};
        auto rows = std::make_shared<TensileLite::ContractionProblemSelectionLibrary>();
        rows->rows.emplace_back(
            TensileLite::ContractionProblemPredicate(
                std::make_shared<TensileLite::Predicates::And<Gemm>>(preds)),
            equality);
        master->library = rows;
        return std::dynamic_pointer_cast<Master>(TensileLite::LoadLibraryData<Gemm, Solution>(
            TensileLite::MessagePackWriteLibraryData(*master)));
    }

    TensileLite::hip::HipAMDGPU device(std::string const& amdgpuVersion, long long xgmiPhysicalId)
    {
        TensileLite::hip::HipAMDGPU gpu;
        gpu.processor        = TensileLite::AMDGPU::Processor::gfx950;
        gpu.computeUnitCount = 256;
        gpu.deviceName       = "AMD Instinct MI355X";
        gpu.amdgpuVersion    = amdgpuVersion;
        gpu.xgmiPhysicalId   = xgmiPhysicalId;
        return gpu;
    }
}

// amdgpu 6.19 and newer, whatever follows the minor version, have the rule; there the
// bit is xgmi_physical_id & 1, and an unreadable id gives none.
TEST(HostUnitAidPlacement, smoke_DriverRuleAcceptsAmdgpu619AndNewer)
{
    for(auto const& v : kRejected)
    {
        EXPECT_FALSE(page24RuleHolds(v)) << "'" << v << "'";
        EXPECT_EQ(aidPlacementBit(v, 7), std::nullopt) << "'" << v << "'";
        EXPECT_EQ(device(v, 7).aidPlacementBit(), std::nullopt) << "'" << v << "'";
    }
    for(auto const& v : kAccepted)
    {
        EXPECT_TRUE(page24RuleHolds(v)) << "'" << v << "'";
        EXPECT_EQ(aidPlacementBit(v, 7), 1) << "'" << v << "'";
        EXPECT_EQ(aidPlacementBit(v, 6), 0) << "'" << v << "'";
        EXPECT_EQ(aidPlacementBit(v, 0), 0) << "'" << v << "'";
        EXPECT_EQ(aidPlacementBit(v, -1), std::nullopt) << "'" << v << "'";
        EXPECT_EQ(device(v, 5).aidPlacementBit(), 1) << "'" << v << "'";
    }
}

// Selection skips a kernel that takes the bit on a driver without the rule, and on a device
// whose xgmi_physical_id is unknown, so the GEMM finds no solution rather than one it would
// launch with an invented bit. The same kernel without the argument is selected on any driver.
TEST(HostUnitAidPlacement, smoke_SelectionRejectsTheKernelOnOtherDrivers)
{
    auto const aid   = page24Library(true);
    auto const plain = page24Library(false);
    ASSERT_NE(aid, nullptr);
    ASSERT_NE(plain, nullptr);
    ASSERT_EQ(aid->solutions.size(), 1u);
    auto const& args = aid->solutions.at(0)->customKernel.args;
    ASSERT_FALSE(args.empty());
    EXPECT_EQ(args.back().semantic, TensileLite::CustomArgSemantic::AIDPlacementBit);

    auto const problem = page24Problem();
    for(auto const& v : kRejected)
    {
        auto const gpu = device(v, 7);
        EXPECT_EQ(aid->findBestSolution(problem, gpu), nullptr) << "amdgpu '" << v << "'";
        EXPECT_FALSE(aid->solutions.at(0)->aidPlacementSupported(gpu)) << "amdgpu '" << v << "'";
        EXPECT_NE(plain->findBestSolution(problem, gpu), nullptr) << "amdgpu '" << v << "'";
    }
    // The selection cache keys on the processor, CU count and name, not on the driver or
    // the xgmi id, so a device that must reject the kernel is asked of a fresh library.
    for(auto const& v : kAccepted)
    {
        EXPECT_EQ(page24Library(true)->findBestSolution(problem, device(v, -1)), nullptr)
            << "amdgpu '" << v << "', no xgmi_physical_id";
        auto const selected = page24Library(true)->findBestSolution(problem, device(v, 7));
        ASSERT_NE(selected, nullptr) << "amdgpu '" << v << "'";
        EXPECT_EQ(selected->index, 0);
    }

    // Hardware that is not a HIP device has no driver to ask.
    TensileLite::AMDGPU plainGpu(TensileLite::AMDGPU::Processor::gfx950, 256, "AMD Instinct MI355X");
    EXPECT_EQ(page24Library(true)->findBestSolution(problem, plainGpu), nullptr);
    EXPECT_NE(page24Library(false)->findBestSolution(problem, plainGpu), nullptr);
}
