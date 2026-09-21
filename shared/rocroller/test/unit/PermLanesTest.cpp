// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/TensorDescriptor.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace PermLanesTest
{
    struct PermLanesTest : public CurrentGPUContextFixture
    {
    };

    void executePermLanesBlockScale(rocRoller::ContextPtr context,
                                    const int             waveMN,
                                    const int             waveK,
                                    const int             miMN,
                                    const int             miK)
    {
        int  MN     = waveMN * 4;
        int  K      = waveK;
        auto unit   = Expression::literal(1);
        auto exprMN = Expression::literal(MN);
        auto exprK  = Expression::literal(K);

        int macMN = MN;
        int macK  = K;
        int waveB = 1;
        int miB   = 1;

        rocRoller::KernelGraph::KernelGraph kgraph;

        auto kernel = kgraph.control.addElement(Kernel());
        auto load   = kgraph.control.addElement(LoadTiled(DataType::E8M0));
        kgraph.control.addElement(Body(), {kernel}, {load});
        auto exchange = kgraph.control.addElement(Exchange(DataType::E8M0));
        kgraph.control.addElement(Sequence(), {load}, {exchange});
        auto store = kgraph.control.addElement(StoreTiled(DataType::E8M0));
        kgraph.control.addElement(Sequence(), {exchange}, {store});

        auto user0 = kgraph.coordinates.addElement(
            User("a", std::make_shared<Expression::Expression>((size_t)MN * K)));
        auto idim0 = kgraph.coordinates.addElement(SubDimension(0, exprMN, exprK));
        auto idim1 = kgraph.coordinates.addElement(SubDimension(1, exprK, unit));
        kgraph.coordinates.addElement(Split(), {user0}, {idim0, idim1});
        auto mactile0 = kgraph.coordinates.addElement(MacroTile({macMN, macK},
                                                                LayoutType::MATRIX_A,
                                                                {waveMN, waveMN, waveK, waveB},
                                                                MemoryType::WAVE_SWIZZLE,
                                                                {miMN, miMN, miK, miB}));
        kgraph.coordinates.addElement(ConstructMacroTile(), {idim0, idim1}, {mactile0});
        kgraph.coordinates.addElement(DataFlow(), {user0}, {mactile0});
        kgraph.mapper.connect<User>(load, user0);
        kgraph.mapper.connect<MacroTile>(load, mactile0);
        kgraph.mapper.connect<MacroTile>(exchange, mactile0);

        auto user1 = kgraph.coordinates.addElement(
            User("result", std::make_shared<Expression::Expression>((size_t)MN * K)));
        auto odim0    = kgraph.coordinates.addElement(SubDimension(0, exprMN, exprK));
        auto odim1    = kgraph.coordinates.addElement(SubDimension(1, exprK, unit));
        auto mactile1 = kgraph.coordinates.addElement(MacroTile({macMN, macK},
                                                                LayoutType::MATRIX_A,
                                                                {waveMN, waveMN, waveK, waveB},
                                                                MemoryType::WAVE_SWIZZLE,
                                                                {miMN, miMN, miK, miB}));
        kgraph.coordinates.addElement(DestructMacroTile(), {mactile1}, {odim0, odim1});
        kgraph.coordinates.addElement(Join(), {odim0, odim1}, {user1});
        kgraph.coordinates.addElement(DataFlow(), {mactile1}, {user1});

        kgraph.mapper.connect<MacroTile>(store, mactile1);
        kgraph.mapper.connect(exchange, mactile1, NaryArgument::DEST);
        kgraph.mapper.connect<User>(store, user1);

        auto k = context->kernel();

        k->setKernelDimensions(2);
        auto one               = Expression::literal(1u);
        auto workitemCountExpr = Expression::literal(256u);
        k->setWorkitemCount({workitemCountExpr, one, one});
        k->setWorkgroupSize({256, 1, 1});

        auto params = std::make_shared<CommandParameters>();
        params->setWaveTilesPerWavefront(1, 1);
        params->setManualWavefrontCount({4, 1});
        auto lowerTile             = std::make_shared<LowerTile>(params, context);
        kgraph                     = kgraph.transform(lowerTile);
        auto updateWavefrontParams = std::make_shared<UpdateWavefrontParameters>(params);
        kgraph                     = kgraph.transform(updateWavefrontParams);

        auto command = std::make_shared<rocRoller::Command>();
        command->allocateArgument({DataType::E8M0, PointerType::PointerGlobal},
                                  Operations::OperationTag(0),
                                  ArgumentType::Value,
                                  DataDirection::WriteOnly,
                                  "result");
        command->allocateArgument({DataType::E8M0, PointerType::PointerGlobal},
                                  Operations::OperationTag(1),
                                  ArgumentType::Value,
                                  DataDirection::ReadOnly,
                                  "a");
        auto assignIndexExprs = std::make_shared<AssignIndexExpressions>(context, command);
        kgraph                = kgraph.transform(assignIndexExprs);

        kgraph = kgraph.transform(std::make_shared<LoadPacked>(context));
        if(context->kernelOptions()->removeSetCoordinate)
            kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        kgraph = kgraph.transform(std::make_shared<CleanArguments>(context, command));

        context->schedule(k->preamble());
        context->schedule(k->prolog());
        context->schedule(rocRoller::KernelGraph::generate(kgraph, context->kernel()));

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        std::vector<uint8_t> result(MN * K, 0);

        auto now = static_cast<uint32_t>(std::time(0));
        std::cout << "seed: " << now << std::endl;
        RandomGenerator random(now);

        auto a = random.vector<uint8_t>(MN * K, 0, 9);

        auto dA      = make_shared_device<uint8_t>(a);
        auto dResult = make_shared_device<uint8_t>(result);

        KernelArguments kargs;
        kargs.append("a", dA.get());
        kargs.append("result", dResult.get());

        KernelInvocation kinv;
        kinv.workitemCount    = {256, 1, 1};
        kinv.workgroupSize    = {256, 1, 1};
        auto executableKernel = context->instructions()->getExecutableKernel();
        executableKernel->executeKernel(kargs, kinv);

        ASSERT_THAT(
            hipMemcpy(
                result.data(), dResult.get(), result.size() * sizeof(uint8_t), hipMemcpyDefault),
            HasHipSuccess(0));

        int nWaves          = 4;
        int nLanes          = 16;
        int nSIMDsPerWave   = 4;
        int nSIMDIndex      = waveMN / nLanes;
        int nSIMDBlock      = nSIMDsPerWave / nSIMDIndex;
        int nVGPRIndex      = std::min(nSIMDIndex, miK);
        int nVGPRBlock      = waveK / nSIMDBlock / nVGPRIndex;
        int nSIMDIndexBlock = nVGPRIndex;
        int nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

        // clang-format off
        for(int wave      = 0;      wave < nWaves; wave++)
        for(int simdIndexBlock = 0; simdIndexBlock < nSIMDIndexBlock; simdIndexBlock++)
        for(int simdIndexIndex = 0; simdIndexIndex < nSIMDIndexIndex; simdIndexIndex++)
        for(int lane      = 0;      lane < nLanes; lane++)
        for(int simdBlock = 0; simdBlock < nSIMDBlock; simdBlock++)
        for(int vgprBlock = 0; vgprBlock < nVGPRBlock; vgprBlock++)
        for(int vgprIndex = 0; vgprIndex < nVGPRIndex;    vgprIndex++)
        {
            auto aIdx = wave * nSIMDIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + lane * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdBlock * nVGPRBlock * nVGPRIndex
                        + vgprBlock * nVGPRIndex + vgprIndex;

            auto resultIdx = aIdx;

            if(waveMN == 64)
            {
                resultIdx = wave * nSIMDIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + vgprIndex * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + lane * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdBlock * nVGPRBlock * nVGPRIndex
                        + vgprBlock * nVGPRIndex + simdIndexBlock;
            }
            else if(waveMN == 32 && miMN == 16)
            {
                resultIdx = wave * nSIMDIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + vgprIndex * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + lane * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + vgprBlock * nVGPRBlock * nVGPRIndex
                        + simdBlock * nVGPRIndex + simdIndexBlock;
            }
            else if(waveMN == 32 && miMN == 32)
            {
                resultIdx = wave * nSIMDIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexBlock * nSIMDIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + simdIndexIndex * nLanes * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + lane * nSIMDBlock * nVGPRBlock * nVGPRIndex
                        + vgprIndex * nVGPRBlock * nVGPRIndex
                        + simdBlock * nVGPRIndex + vgprBlock;
            }

            ASSERT_EQ(a[aIdx], result[resultIdx]);
        }
        // clang-format on

        std::vector<size_t> sizes = {static_cast<size_t>(nVGPRIndex),
                                     static_cast<size_t>(nVGPRBlock),
                                     static_cast<size_t>(nSIMDBlock),
                                     static_cast<size_t>(nLanes),
                                     static_cast<size_t>(nSIMDIndexIndex),
                                     static_cast<size_t>(nSIMDIndexBlock),
                                     static_cast<size_t>(nWaves)};

        TensorDescriptor src(DataType::E8M0, sizes), dst;
        if(waveMN == 64)
            dst = TensorDescriptor::ShuffledNoPadding(DataType::E8M0, sizes, {5, 1, 2, 3, 4, 0, 6});
        if(waveMN == 32 && miMN == 16)
            dst = TensorDescriptor::ShuffledNoPadding(DataType::E8M0, sizes, {5, 2, 1, 3, 4, 0, 6});
        if(waveMN == 32 && miMN == 32)
            dst = TensorDescriptor::ShuffledNoPadding(DataType::E8M0, sizes, {1, 2, 0, 3, 4, 5, 6});

        auto a_reordered = shuffleDims(a, dst, src);
        EXPECT_EQ(a_reordered, result);
    }

    TEST_F(PermLanesTest, PermLanesBlockScale64x4MI16x4GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes16);
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 64, 4, 16, 4);
    }

    TEST_F(PermLanesTest, PermLanesBlockScale32x8MI16x4GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes16);
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 32, 8, 16, 4);
    }

    TEST_F(PermLanesTest, PermLanesBlockScale64x4MI32x2GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 64, 4, 32, 2);
    }

    TEST_F(PermLanesTest, PermLanesBlockScale32x8MI32x2GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 32, 8, 32, 2);
    }
}
