/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

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

    void executePermLanesBlockScale(rocRoller::ContextPtr context, const int miMN, const int miK)
    {
        int  MN     = 256;
        int  K      = 4;
        auto unit   = Expression::literal(1);
        auto exprMN = Expression::literal(256);
        auto exprK  = Expression::literal(4);

        int macMN  = 256;
        int macK   = 4;
        int waveMN = 64;
        int waveK  = 4;
        int waveB  = 1;
        int miB    = 1;

        rocRoller::KernelGraph::KernelGraph kgraph;

        auto kernel = kgraph.control.addElement(Kernel());
        auto load   = kgraph.control.addElement(LoadTiled(DataType::UInt8));
        kgraph.control.addElement(Body(), {kernel}, {load});
        auto exchange = kgraph.control.addElement(Exchange(DataType::UInt8));
        kgraph.control.addElement(Sequence(), {load}, {exchange});
        auto store = kgraph.control.addElement(StoreTiled(DataType::UInt8));
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
        auto odim0 = kgraph.coordinates.addElement(SubDimension(0, exprMN, exprK));
        auto odim1 = kgraph.coordinates.addElement(SubDimension(1, exprK, unit));
        auto mactile1
            = kgraph.coordinates.addElement(MacroTile({macMN, macK}, MemoryType::VGPR, {1, 4}));
        kgraph.coordinates.addElement(DestructMacroTile(), {mactile1}, {odim0, odim1});
        kgraph.coordinates.addElement(Join(), {odim0, odim1}, {user1});
        kgraph.coordinates.addElement(DataFlow(), {mactile1}, {user1});

        kgraph.mapper.connect<MacroTile>(store, mactile1);
        kgraph.mapper.connect(exchange, mactile1, NaryArgument::DEST);
        kgraph.mapper.connect<User>(store, user1);

        auto k = context->kernel();

        k->addArgument(
            {"a", {DataType::UInt8, PointerType::PointerGlobal}, DataDirection::ReadOnly});
        k->addArgument(
            {"result", {DataType::UInt8, PointerType::PointerGlobal}, DataDirection::WriteOnly});

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
        auto addComputeIndex       = std::make_shared<AddComputeIndex>();
        kgraph                     = kgraph.transform(addComputeIndex);
        auto updateWavefrontParams = std::make_shared<UpdateWavefrontParameters>(params);
        kgraph                     = kgraph.transform(updateWavefrontParams);

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

        auto nWaves = 4;
        auto factor = waveK / miK;
        auto nLanes = 16;
        for(int wave = 0; wave < nWaves; wave++)
            for(int simdBlock = 0; simdBlock < miK; simdBlock++)
                for(int simdIndex = 0; simdIndex < factor; simdIndex++)
                    for(int lane = 0; lane < nLanes; lane++)
                        for(int vgprBlock = 0; vgprBlock < factor; vgprBlock++)
                            for(int vgprIndex = 0; vgprIndex < miK; vgprIndex++)
                            {
                                ASSERT_EQ(a[wave * waveK * nLanes * waveK
                                            + simdBlock * factor * nLanes * waveK
                                            + simdIndex * nLanes * waveK + lane * waveK
                                            + vgprBlock * miK + vgprIndex],
                                          result[wave * waveK * nLanes * waveK
                                                 + vgprIndex * factor * nLanes * waveK
                                                 + simdIndex * nLanes * waveK + lane * waveK
                                                 + vgprBlock * miK + simdBlock]);
                            }
    }

    TEST_F(PermLanesTest, PermLanesBlockScale16x4GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes16);
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 16, 4);
    }

    TEST_F(PermLanesTest, PermLanesBlockScale32x2GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasPermLanes32);
        executePermLanesBlockScale(m_context, 32, 2);
    }
}
