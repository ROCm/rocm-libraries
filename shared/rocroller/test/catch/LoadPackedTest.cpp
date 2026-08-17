// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <common/CommonGraphs.hpp>

TEST_CASE("LoadPacked", "[kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;

    auto context = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::TileDoubleAdd<Half>();

    example.setTileSize(16, 8);
    example.setSubTileSize(4, 2);

    auto params = example.getCommandParameters(512, 512);
    auto graph  = example.getKernelGraph();

    std::vector<GraphTransformPtr> transforms{
        std::make_shared<UpdateParameters>(params),
        std::make_shared<AddLDS>(params, context.get()),
        std::make_shared<LowerLinear>(context.get()),
        std::make_shared<LowerTile>(params, context.get()),
        std::make_shared<AssignIndexExpressions>(context.get(), example.getCommand()),
        std::make_shared<UpdateWavefrontParameters>(params),
    };

    for(auto const& xform : transforms)
        graph = graph.transform(xform);

    auto verifyLoadStoreOpVarType = [&graph](Operation op, VariableType variableType) {
        std::visit(
            [&](auto&& op) {
                using T = std::decay_t<decltype(op)>;

                if constexpr(CIsAnyOf<T, LoadTiled, StoreTiled, LoadLDSTile, StoreLDSTile>)
                {
                    for(auto opTag : graph.control.getElements<T>())
                    {
                        auto operation = graph.control.get<T>(opTag).value();
                        CHECK(operation.varType == variableType);
                    }
                }
            },
            op);
    };

    auto variableType       = DataType::Half;
    auto packedVariableType = DataTypeInfo::Get(variableType).packedVariableType().value();
    verifyLoadStoreOpVarType(LoadTiled{}, variableType);
    verifyLoadStoreOpVarType(StoreTiled{}, variableType);
    verifyLoadStoreOpVarType(LoadLDSTile{}, variableType);
    verifyLoadStoreOpVarType(StoreLDSTile{}, variableType);

    graph = graph.transform(std::make_shared<LoadPacked>(context.get()));

    // After LoadPacked, the {Load | Store}LDSTile::varType should be
    // the corresponding packed type. The LoadPacked analysis should not
    // identify the {Load | Store}Tiled operations as applicable for the
    // transformation in this case, and thus the varType should remain
    // unchanged.
    verifyLoadStoreOpVarType(LoadTiled{}, variableType);
    verifyLoadStoreOpVarType(StoreTiled{}, variableType);
    verifyLoadStoreOpVarType(LoadLDSTile{}, packedVariableType);
    verifyLoadStoreOpVarType(StoreLDSTile{}, packedVariableType);
}
