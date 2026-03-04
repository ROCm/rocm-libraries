// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/Utilities.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace Direct2LDSTileCopyTest
{
    void addDirect2LDS(rocRoller::KernelGraph::KernelGraph& kgraph, DataType dt)
    {
        auto loadTiledNodes    = kgraph.control.getNodes<LoadTiled>().to<std::vector>();
        auto storeLDSTileNodes = kgraph.control.getNodes<StoreLDSTile>().to<std::vector>();
        for(auto loadGlobal : loadTiledNodes)
        {
            auto internalMacroTile = kgraph.mapper.get<MacroTile>(loadGlobal);
            for(auto storeLDS : storeLDSTileNodes)
            {
                if(kgraph.mapper.get<MacroTile>(storeLDS) == internalMacroTile)
                {
                    auto direct2lds = kgraph.control.addElement(LoadTileDirect2LDS(dt));

                    moveConnections(kgraph, loadGlobal, direct2lds, 0);
                    moveConnections(kgraph, storeLDS, direct2lds, 2);

                    replaceWith(kgraph, loadGlobal, direct2lds, false);
                    replaceWith(kgraph, storeLDS, kgraph.control.addElement(NOP()), false);

                    purgeNodes(kgraph, {loadGlobal, storeLDS});
                }
            }
        }
    }

    TEMPLATE_TEST_CASE("Direct2LDS tile copy", "[direct2lds][gpu]", uint8_t, uint16_t, uint32_t)
    {
        using T = TestType;
        auto dt = TypeInfo<T>::Var.dataType;

        auto testContext = TestContext::ForTestDevice();
        auto context     = testContext.get();

        if(!context->targetArchitecture().HasCapability(GPUCapability::HasDirectToLds))
        {
            SKIP("Architecture " + context->targetArchitecture().target().toString()
                 + " does not support DirectToLDS");
        }

        // Use 64 workitems (1 wave) so the M0+TID*4 stride pattern
        // is deterministic without multi-wave overlap in LDS.
        const int MN = 64;
        const int K  = 1;

        auto expr1u = Expression::literal(1u);
        auto exprMN = Expression::literal(MN);
        auto exprK  = Expression::literal(K);

        int macMN     = MN;
        int macK      = K;
        int subtileMN = 1;
        int subtileK  = 1;

        rocRoller::KernelGraph::KernelGraph kgraph;

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

        kgraph.coordinates.addElement(Split(), {user0}, {idim0, idim1});
        kgraph.coordinates.addElement(ConstructMacroTile(), {idim0, idim1}, {mactile0});
        kgraph.coordinates.addElement(DataFlow(), {mactile0}, {mactile1});
        kgraph.coordinates.addElement(DestructMacroTile(), {mactile1}, {odim0, odim1});
        kgraph.coordinates.addElement(Join(), {odim0, odim1}, {user1});
        kgraph.coordinates.addElement(DataFlow(), {user0}, {mactile0});
        kgraph.coordinates.addElement(DataFlow(), {mactile1}, {user1});

        auto DF = [dt](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Vector, dt});
        };

        auto kernel     = kgraph.control.addElement(Kernel());
        auto load       = kgraph.control.addElement(LoadTiled(dt));
        auto assignCopy = kgraph.control.addElement(Assign{Register::Type::Vector, DF(mactile0)});
        auto store      = kgraph.control.addElement(StoreTiled(dt));

        kgraph.control.addElement(Body(), {kernel}, {load});
        kgraph.control.addElement(Sequence(), {load}, {assignCopy});
        kgraph.control.addElement(Sequence(), {load}, {assignCopy});
        kgraph.control.addElement(Sequence(), {assignCopy}, {store});

        kgraph.mapper.connect<User>(load, user0);
        kgraph.mapper.connect<MacroTile>(load, mactile0);
        kgraph.mapper.connect<MacroTile>(store, mactile1);
        kgraph.mapper.connect<User>(store, user1);
        kgraph.mapper.connect(assignCopy, mactile1, NaryArgument::DEST);

        auto k = context->kernel();

        k->setKernelDimensions(2);
        auto one = Expression::literal(1u);
        auto workitemCountExpr
            = Expression::literal(static_cast<unsigned int>(MN * K) / (subtileMN * subtileK));
        k->setWorkitemCount({workitemCountExpr, one, one});
        k->setWorkgroupSize({static_cast<unsigned int>(MN * K) / (subtileMN * subtileK), 1, 1});

        auto params = std::make_shared<CommandParameters>();
        params->setWaveTilesPerWavefront(1, 1);

        using namespace rocRoller::KernelGraph;

        auto xformAddLDS = std::make_shared<AddLDS>(params, context);
        kgraph           = kgraph.transform(xformAddLDS);

        auto xformAddPrefetch = std::make_shared<AddPrefetch>(params, context);
        kgraph                = kgraph.transform(xformAddPrefetch);

        auto xformLowerTile = std::make_shared<LowerTile>(params, context);
        kgraph              = kgraph.transform(xformLowerTile);

        addDirect2LDS(kgraph, dt);

        auto xformUpdateWavefront = std::make_shared<UpdateWavefrontParameters>(params);
        kgraph                    = kgraph.transform(xformUpdateWavefront);

        kgraph = kgraph.transform(std::make_shared<AddLDSBarriers>());

        auto command = std::make_shared<rocRoller::Command>();
        command->allocateArgument({dt, PointerType::PointerGlobal},
                                  Operations::OperationTag(0),
                                  ArgumentType::Value,
                                  DataDirection::WriteOnly,
                                  "result");
        command->allocateArgument({dt, PointerType::PointerGlobal},
                                  Operations::OperationTag(1),
                                  ArgumentType::Value,
                                  DataDirection::ReadOnly,
                                  "a");
        auto assignIndexExprs = std::make_shared<AssignIndexExpressions>(context, command);
        kgraph                = kgraph.transform(assignIndexExprs);

        if(context->kernelOptions()->removeSetCoordinate)
            kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        kgraph = kgraph.transform(std::make_shared<CleanArguments>(context, command));

        context->schedule(k->preamble());
        context->schedule(k->prolog());
        context->schedule(rocRoller::KernelGraph::generate(kgraph, context->kernel()));
        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        auto code = context->instructions()->toString();

        std::string expectedInstr = (sizeof(T) == 1)   ? "buffer_load_ubyte"
                                    : (sizeof(T) == 2) ? "buffer_load_ushort"
                                                       : "buffer_load_dword";
        CHECK(countSubstring(code, expectedInstr) == 1);

        std::vector<char> assembledKernel = context->instructions()->assemble();
        REQUIRE(assembledKernel.size() > 0);

        std::vector<T> a(MN * K);
        for(size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<T>(i);

        auto dA      = make_shared_device(a);
        auto dResult = make_shared_device<T>(MN * K);

        KernelArguments kargs;
        kargs.append("a", dA.get());
        kargs.append("result", dResult.get());

        KernelInvocation kinv;
        kinv.workitemCount    = {static_cast<unsigned int>(MN), 1, 1};
        kinv.workgroupSize    = {static_cast<unsigned int>(MN), 1, 1};
        auto executableKernel = context->instructions()->getExecutableKernel();
        executableKernel->executeKernel(kargs, kinv);

        std::vector<T> expected(MN * K);

        if(sizeof(T) < 4)
        {
            // Direct2LDS writes at M0 + TID*4 (always dword-aligned).
            // For sub-dword types this leaves zero gaps in LDS:
            //   uint8:  [0,0,0,0, 1,0,0,0, 2,0,0,0, ...]  (stride 4)
            //   uint16: [0,0, 1,0, 2,0, 3,0, ...]          (stride 2)
            constexpr int stride = 4 / sizeof(T);
            for(int i = 0; i < MN * K; ++i)
                expected[i] = (i % stride == 0) ? static_cast<T>(i / stride) : T(0);
        }
        else
        {
            for(int i = 0; i < MN * K; ++i)
                expected[i] = static_cast<T>(i);
        }

        REQUIRE_THAT(dResult, HasDeviceVectorEqualTo(expected));
    }
}
