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
#include <rocRoller/KernelGraph/Utils.hpp>

#include "GPUContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace LDSCopyTest
{
    struct LDSCopyTest : public GPUContextFixture
    {
    };

    // TODO: make it more general and works for GEMM problem as a graph transform
    /*
     * This function replaces a pair of {LoadTiled, StoreLDSTile} with a {LoadTileDirect2LDS}
     * An example to illustrate:
     *
     * Original control graph:
     *
     * Kernel
     *   |
     *   --[body]--> Scope --[body]--> ComputeIndex --[seq]--> ComputeIndex --[seq]--> LoadTiled
     *                 |
     *                 |[seq]
     *                 |
     *                 v
     *              Barrier --[seq]--> Scope --[seq]--> ComputeIndex --[seq]--> ComputeIndex --[seq] -->StoreLDSTiled
     *
     *
     * New control graph:
     *
     * Kernel
     *   |
     *   --[body]--> Scope --[body]--> ComputeIndex --[seq]--> ComputeIndex --[seq]--> NOP
     *                                                                                  |
     *                                                                                  |[seq]
     *                                                                                  |
     *                                                                                  v
     *                                                                             ComputeIndex
     *                                                                                  |
     *                                                                                  |[seq]
     *                                                                                  |
     *                                                                                  v
     *                                                                             ComputeIndex
     *                                                                                  |
     *                                                                                  |[seq]
     *                                                                                  |
     *                                                                                  v
     *                                                                                 NOP
     *                                                                                  |
     *                                                                                  |[seq]
     *                                                                                  |
     *                                                                                  v
     *                                                                                Barrier
     *                                                                                  |
     *                                                                                  |[seq]
     *                                                                                  |
     *                                                                                  v
     *                                                                          LoadTileDirect2LDS
    */
    void addDirect2LDS(rocRoller::KernelGraph::KernelGraph& kgraph)
    {
        auto loadTiledNodes    = kgraph.control.getNodes<LoadTiled>().to<std::vector>();
        auto storeLDSTileNodes = kgraph.control.getNodes<StoreLDSTile>().to<std::vector>();

        AssertFatal(loadTiledNodes.size() == storeLDSTileNodes.size());

        for(auto loadGlobal : loadTiledNodes)
        {
            auto globalOpChildren
                = kgraph.control.getNeighbours<Graph::Direction::Downstream>(loadGlobal)
                      .template to<std::vector>();
            AssertFatal(globalOpChildren.empty());

            auto internalMacroTile = kgraph.mapper.get<MacroTile>(loadGlobal);

            for(auto storeLDS : storeLDSTileNodes)
            {
                // find the pair of LoadTiled and StoreLDSTile operations
                if(kgraph.mapper.get<MacroTile>(storeLDS) == internalMacroTile)
                {
                    // find the barrier before StoreLDSTile operation
                    int lastTag         = -1;
                    int computeIndexTag = -1;
                    for(auto parent :
                        kgraph.control.depthFirstVisit(storeLDS, Graph::Direction::Upstream))
                    {
                        bool containing = lastTag != -1
                                          && (kgraph.control.get<Body>(lastTag)
                                              || kgraph.control.get<Sequence>(lastTag));
                        lastTag = parent;
                        if(kgraph.control.get<ComputeIndex>(parent))
                        {
                            computeIndexTag = parent;
                        }

                        auto barrier = kgraph.control.get<Barrier>(parent);
                        if(barrier && containing)
                        {
                            break;
                        }
                    }

                    // add LoadTileDirect2LDS operation
                    auto direct2lds
                        = kgraph.control.addElement(LoadTileDirect2LDS(DataType::UInt32));
                    auto barrier = kgraph.control.addElement(Barrier());
                    kgraph.control.addElement(Sequence(), {barrier}, {direct2lds});

                    // move the LoadTiled and StoreLDSTile connections to Direct2LDS
                    moveConnections(kgraph, loadGlobal, direct2lds);
                    moveConnections(kgraph, storeLDS, direct2lds);

                    // merge LoadTiled and StoreLDSTile operations
                    AssertFatal(computeIndexTag != -1);
                    reconnect<Graph::Direction::Upstream>(kgraph, -1, computeIndexTag);
                    kgraph.control.addElement(Sequence(), {loadGlobal}, {computeIndexTag});
                    kgraph.control.addElement(Sequence(), {storeLDS}, {barrier});

                    replaceWith(kgraph, loadGlobal, kgraph.control.addElement(NOP()), false);
                    replaceWith(kgraph, storeLDS, kgraph.control.addElement(NOP()), false);
                    replaceWith(kgraph, lastTag, kgraph.control.addElement(NOP()), false);

                    purgeNodes(kgraph, {loadGlobal, storeLDS, lastTag});
                }
            }
        }
    }

    TEST_P(LDSCopyTest, GPU_LDSCopyTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDirectToLds);

        if(!m_context->targetArchitecture().target().isCDNAGPU())
            GTEST_SKIP() << "Skipping LDS tile add tests for " << GetParam();

        const int MN = 256;
        const int K  = 1;

        auto expr1u = Expression::literal(1u);
        auto exprMN = Expression::literal(MN);
        auto exprK  = Expression::literal(K);

        int macMN     = MN;
        int macK      = K;
        int subtileMN = 1;
        int subtileK  = 1;

        rocRoller::KernelGraph::KernelGraph kgraph;

        //
        // Coordinate graph (tile copy) looks like:
        //                            (User0)
        //                         /         \
        //                 [Split]           [DataFlow]
        //                 /     \              |
        //                v       v             |
        //     (Subdimension0) (Subdimension1)  |
        //               \         /            |
        //           [ConstructMacroTile]       |
        //                    |                 |
        //                    v                 v
        //                       (MacroTile0:LDS)
        //                             |
        //                          [DataFlow]
        //                             |
        //                             v
        //                       (MacroTile1:VGPR)
        //                       /          \
        //             [Deconstruct]       [DataFlow]
        //              /         \             |
        //             v           v            |
        // (Subdimension0)  (Subdimension1)     |
        //             \         /              |
        //               [Join]                 |
        //                  |                   |
        //                  v                   v
        //                         (User1)

        // coordinate nodes
        auto user0 = kgraph.coordinates.addElement(
            User("a", std::make_shared<Expression::Expression>((size_t)MN * K)));
        auto idim0    = kgraph.coordinates.addElement(SubDimension(0, exprMN, exprK));
        auto idim1    = kgraph.coordinates.addElement(SubDimension(1, exprK, expr1u));
        auto mactile0 = kgraph.coordinates.addElement(
            MacroTile({macMN, macK}, MemoryType::LDS, {subtileMN, subtileK}));

        auto mactile1 = kgraph.coordinates.addElement(
            MacroTile({macMN, macK}, MemoryType::VGPR, {subtileMN, subtileK}));

        auto odim0 = kgraph.coordinates.addElement(SubDimension(0, exprMN, exprK));
        auto odim1 = kgraph.coordinates.addElement(SubDimension(1, exprK, expr1u));
        auto user1 = kgraph.coordinates.addElement(
            User("result", std::make_shared<Expression::Expression>((size_t)MN * K)));

        // coordinate edges
        kgraph.coordinates.addElement(Split(), {user0}, {idim0, idim1});
        kgraph.coordinates.addElement(ConstructMacroTile(), {idim0, idim1}, {mactile0});
        kgraph.coordinates.addElement(DataFlow(), {mactile0}, {mactile1});
        kgraph.coordinates.addElement(DestructMacroTile(), {mactile1}, {odim0, odim1});
        kgraph.coordinates.addElement(Join(), {odim0, odim1}, {user1});
        kgraph.coordinates.addElement(DataFlow(), {user0}, {mactile0});
        kgraph.coordinates.addElement(DataFlow(), {mactile1}, {user1});

        //
        // Control graph looks like:
        //         (Kernel)
        //            |
        //          [Body]
        //            |
        //            v
        //       (LoadTiled)
        //          /     \
        //   [Sequence]   [Sequence]
        //       |            |
        //       v            v
        // (Assign VGPR Add(DataFlowTag0, DataFlowTag0))
        //            |
        //         [Sequence]
        //            |
        //            v
        //       (StoreTiled)

        auto DF = [](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Vector, DataType::UInt32});
        };

        auto kernel = kgraph.control.addElement(Kernel());
        auto load   = kgraph.control.addElement(LoadTiled(DataType::UInt32));

        auto assignAdd = kgraph.control.addElement(
            Assign{Register::Type::Vector, DF(mactile0) + DF(mactile0)});
        auto store = kgraph.control.addElement(StoreTiled(DataType::UInt32));

        auto body      = kgraph.control.addElement(Body(), {kernel}, {load});
        auto sequence1 = kgraph.control.addElement(Sequence(), {load}, {assignAdd});
        auto sequence2 = kgraph.control.addElement(Sequence(), {load}, {assignAdd});
        kgraph.control.addElement(Sequence(), {assignAdd}, {store});

        // connect
        kgraph.mapper.connect<User>(load, user0);
        kgraph.mapper.connect<MacroTile>(load, mactile0);
        kgraph.mapper.connect<MacroTile>(store, mactile1);
        kgraph.mapper.connect<User>(store, user1);
        kgraph.mapper.connect(assignAdd, mactile1, NaryArgument::DEST);

        auto k = m_context->kernel();

        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly});
        k->addArgument(
            {"result", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(2);
        auto one = Expression::literal(1u);
        auto workitemCountExpr
            = Expression::literal(static_cast<unsigned int>(MN * K) / (subtileMN * subtileK));
        k->setWorkitemCount({workitemCountExpr, one, one});
        k->setWorkgroupSize({static_cast<unsigned int>(MN * K) / (subtileMN * subtileK), 1, 1});

        auto params = std::make_shared<CommandParameters>();
        params->setWaveTilesPerWavefront(1, 1);

        auto addLDS = std::make_shared<AddLDS>(params, m_context);
        kgraph      = kgraph.transform(addLDS);

        // adds barriers
        auto addPrefetch = std::make_shared<AddPrefetch>(params, m_context);
        kgraph           = kgraph.transform(addPrefetch);

        auto lowerTile = std::make_shared<LowerTile>(params, m_context);
        kgraph         = kgraph.transform(lowerTile);

        auto addComputeIndex = std::make_shared<AddComputeIndex>();
        kgraph               = kgraph.transform(addComputeIndex);

        // manually add LoadTileDirect2LDS operation
        addDirect2LDS(kgraph);

        auto updateWavefrontParams = std::make_shared<UpdateWavefrontParameters>(params);
        kgraph                     = kgraph.transform(updateWavefrontParams);

        // FIXME: this is a workaround as currrently the wait count observer
        //        only emits lgckmcnt(0), while loadToLDS additionally
        //        requires vmcnt(0).
        setKernelOptions({.alwaysWaitZeroBeforeBarrier = 1});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());
        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(!isLocalDevice())
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
        else
        {
            std::vector<uint32_t> a(MN * K, 1);
            std::vector<uint32_t> result(MN * K);
            for(int i = 0; i < MN; i++)
                for(int j = 0; j < K; j++)
                    a[i * K + j] = i * K + j;

            auto dA      = make_shared_device(a);
            auto dResult = make_shared_device(result);

            KernelArguments kargs;
            kargs.append("a", dA.get());
            kargs.append("result", dResult.get());

            KernelInvocation kinv;
            kinv.workitemCount    = {MN, 1, 1};
            kinv.workgroupSize    = {MN, 1, 1};
            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            ASSERT_THAT(hipMemcpy(result.data(),
                                  dResult.get(),
                                  result.size() * sizeof(uint32_t),
                                  hipMemcpyDefault),
                        HasHipSuccess(0));

            for(int i = 0; i < MN * K; i++)
            {
                ASSERT_EQ(a[i] + a[i], result[i]);
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(LDSCopyTest, LDSCopyTest, supportedISATuples());
}
