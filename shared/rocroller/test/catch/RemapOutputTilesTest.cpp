// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups_detail.hpp>
#include <rocRoller/KernelGraph/Transforms/RemapOutputTiles.hpp>
#include <rocRoller/KernelGraph/Transforms/RemapOutputTiles_detail.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "ExpressionMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"
#include "common/Utilities.hpp"

namespace RemapOutputTilesTest
{
    TEST_CASE("TileSizeInfo", "[kernel-graph]")
    {
        using namespace rocRoller::Expression;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::RemapOutputTilesDetail;

        using GD = rocRoller::Graph::Direction;

        KernelGraph graph0;

        int vX = 5, vY = 7;

        auto tileNumAD = graph0.coordinates.addElement(MacroTileNumber(0, literal(vX), nullptr));
        auto tileNumBD = graph0.coordinates.addElement(MacroTileNumber(1, literal(vY), nullptr));

        auto middleLinear = graph0.coordinates.addElement(Linear());

        graph0.coordinates.addElement(Flatten(), {tileNumAD, tileNumBD}, {middleLinear});

        auto tileNumAU = graph0.coordinates.addElement(MacroTileNumber(0, literal(vX), nullptr));
        auto tileNumBU = graph0.coordinates.addElement(MacroTileNumber(1, literal(vY), nullptr));

        graph0.coordinates.addElement(Tile(), {middleLinear}, {tileNumAU, tileNumBU});

        /* coordinate graph is:
         *
         *    MacroTileNumber(0, size=vX)        MacroTileNumber(1, size=vY)
         *             \                                  /
         *              ------------- Flatten ------------
         *                               |
         *                             Linear
         *                               |
         *              --------------- Tile -------------
         *             /                                                \
         *    MacroTileNumber(0, size=vX)        MacroTileNumber(1, size=vY)
         *
         */

        auto info = getTileSizeInfo(graph0);
        CHECK_THAT(info.sizes[0], SimplifiesTo(literal(vX)));
        CHECK_THAT(info.sizes[1], SimplifiesTo(literal(vY)));
        CHECK(info.sizes[2] == nullptr);
        CHECK(workgroupDimensions(info) == 2);

        auto total = totalNumberOfWorkgroups(info);
        CHECK_THAT(total, SimplifiesTo(literal(vX) * literal(vY)));
    }

    class RemapWorkgroupKernel : public AssemblyTestKernel
    {
        using GD = rocRoller::Graph::Direction;

    public:
        RemapWorkgroupKernel(rocRoller::ContextPtr context, int dim)
            : AssemblyTestKernel(context)
            , m_dim(dim)
        {
            makeGraph();
        }

        std::vector<rocRoller::Expression::ExpressionPtr> kernelRemapWorkgroupExpression()
        {
            using namespace rocRoller::KernelGraph::CoordinateGraph;
            using GD = rocRoller::Graph::Direction;

            auto transformer = Transformer(&m_graph->coordinates);
            transformer.fillExecutionCoordinates(m_context);

            auto wgRegister = m_context->registerTagManager()->getRegister(m_workgroupU);
            auto exprs      = m_graph->coordinates.forward(
                {wgRegister->expression()}, {m_workgroupU}, {m_wgx, m_wgy});

            return exprs;
        }

        void generate() override
        {
            using namespace rocRoller;

            auto kernel = m_context->kernel();

            m_context->schedule(kernel->preamble());
            m_context->schedule(kernel->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_wgm, s_wgx, s_wgy;
                co_yield m_context->argLoader()->getValue("WGM", s_wgm);
                co_yield m_context->argLoader()->getValue("WGX", s_wgx);
                co_yield m_context->argLoader()->getValue("WGY", s_wgy);

                auto v_wgx
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);
                auto v_wgy
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::Int32, PointerType::PointerGlobal},
                                                   1);

                co_yield v_wgx->allocate();
                co_yield v_wgy->allocate();
                co_yield m_context->copier()->copy(v_wgx, s_wgx, "Move pointer");
                co_yield m_context->copier()->copy(v_wgy, s_wgy, "Move pointer");
                auto wgIndex = m_context->kernel()->workgroupIndex()[0];
                co_yield Expression::generate(v_wgx,
                                              v_wgx->expression()
                                                  + wgIndex->expression() * Expression::literal(4),
                                              m_context);
                co_yield Expression::generate(v_wgy,
                                              v_wgy->expression()
                                                  + wgIndex->expression() * Expression::literal(4),
                                              m_context);

                Register::ValuePtr s_remappedX, s_remappedY;
                auto               exprs = kernelRemapWorkgroupExpression();
                co_yield Expression::generate(s_remappedX, exprs[0], m_context);
                co_yield Expression::generate(s_remappedY, exprs[1], m_context);

                auto v_remappedX = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {DataType::Int32}, 1);
                co_yield m_context->copier()->copy(v_remappedX, s_remappedX, "Move value");

                auto v_remappedY = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, {DataType::Int32}, 1);
                co_yield m_context->copier()->copy(v_remappedY, s_remappedY, "Move value");

                co_yield m_context->mem()->storeGlobal(
                    v_wgx, v_remappedX, 0, DataTypeInfo::Get(DataType::Int32).elementBytes);
                co_yield m_context->mem()->storeGlobal(
                    v_wgy, v_remappedY, 0, DataTypeInfo::Get(DataType::Int32).elementBytes);
            };

            m_context->schedule(kb());
            m_context->schedule(kernel->postamble());
            m_context->schedule(kernel->amdgpu_metadata());
        }

    private:
        void makeGraph()
        {
            using namespace rocRoller::Expression;
            using namespace rocRoller::KernelGraph;
            using namespace rocRoller::KernelGraph::CoordinateGraph;
            using namespace rocRoller::KernelGraph::RemapOutputTilesDetail;

            auto kernel = m_context->kernel();

            m_command  = std::make_shared<rocRoller::Command>();
            auto wgmOp = m_command->addOperation(
                rocRoller::Operations::Scalar(rocRoller::DataType::Int32));
            auto wgmCommandArgument
                = m_command->allocateArgument(rocRoller::DataType::Int32,
                                              wgmOp,
                                              rocRoller::ArgumentType::Value,
                                              rocRoller::DataDirection::ReadOnly);
            m_wgm = kernel->addArgument({"WGM",
                                         rocRoller::DataType::Int32,
                                         rocRoller::DataDirection::ReadOnly,
                                         wgmCommandArgument->expression()});

            auto numTilesXOp = m_command->addOperation(
                rocRoller::Operations::Scalar(rocRoller::DataType::Int32));
            auto numTilesXCommandArgument
                = m_command->allocateArgument(rocRoller::DataType::Int32,
                                              numTilesXOp,
                                              rocRoller::ArgumentType::Value,
                                              rocRoller::DataDirection::ReadOnly);
            m_numTilesX = kernel->addArgument({"numTilesX",
                                               {rocRoller::DataType::Int32},
                                               rocRoller::DataDirection::ReadOnly,
                                               numTilesXCommandArgument->expression()});

            auto numTilesYOp = m_command->addOperation(
                rocRoller::Operations::Scalar(rocRoller::DataType::Int32));
            auto numTilesYCommandArgument
                = m_command->allocateArgument(rocRoller::DataType::Int32,
                                              numTilesYOp,
                                              rocRoller::ArgumentType::Value,
                                              rocRoller::DataDirection::ReadOnly);
            m_numTilesY = kernel->addArgument({"numTilesY",
                                               {rocRoller::DataType::Int32},
                                               rocRoller::DataDirection::ReadOnly,
                                               numTilesYCommandArgument->expression()});
            enableDivideBy(m_numTilesY, m_context);

            kernel->addArgument(
                {"WGX",
                 {rocRoller::DataType::Int32, rocRoller::PointerType::PointerGlobal},
                 rocRoller::DataDirection::WriteOnly});
            kernel->addArgument(
                {"WGY",
                 {rocRoller::DataType::Int32, rocRoller::PointerType::PointerGlobal},
                 rocRoller::DataDirection::WriteOnly});

            KernelGraph graph;

            TileSizeInfo info{.sizes = {m_numTilesX, m_numTilesY, nullptr}};

            int macroTileNumberTag;

            auto remappedDims = workgroupMapping(
                info, graph, rocRoller::Graph::Direction::Downstream, m_dim, m_wgm);
            macroTileNumberTag = remappedDims.totalTiles;
            m_wgx              = remappedDims.parallelDim;
            m_wgy              = remappedDims.perpendicularDim;

            // Attach workgroup to dangling MacroTileNumber
            ConnectWorkgroupsDetail::connectWorkgroups(graph);

            // Obtain the workgroup attached to the MacroTileNumber
            m_workgroupU = *rocRoller::only(graph.coordinates.getInputNodeIndices(
                macroTileNumberTag, [](auto) { return true; }));

            m_graph = std::make_shared<KernelGraph>(graph);

            kernel->setKernelGraphMeta(m_graph);
        }

        rocRoller::KernelGraph::KernelGraphPtr m_graph;
        rocRoller::CommandPtr                  m_command;

        int m_dim;
        int m_workgroupU;

        std::vector<int> wgs;

        int m_wgx, m_wgy;

        rocRoller::Expression::ExpressionPtr m_wgm, m_numTilesX, m_numTilesY;
    };

    TEST_CASE("Remap Workgroup GPU", "[kernel-graph][gpu]")
    {
        auto remapDim = GENERATE(0, 1);
        {
            // Note:
            //
            //   remapDim = 0 corresponds to the M dimension for D in a GEMM
            //   remapDim = 1 corresponds to the N dimension for D in a GEMM
            //
            // The remapped dimension is baked into the kernel

            auto context = TestContext::ForTestDevice({{.enableFullDivision = true}}, remapDim);
            auto kernel  = RemapWorkgroupKernel(context.get(), remapDim);

            auto numTilesM = GENERATE(22, 55);
            auto numTilesN = GENERATE(7, 8, 11);

            uint totalSize = numTilesM * numTilesN;

            auto WGM = GENERATE(range(1, 50));
            {
                //
                // WGM is the workgroup-mapping "group size".  It is a kernel argument.
                //

                // Launch kernel
                std::vector<int> wgx(totalSize), wgy(totalSize);
                {
                    auto d_wgx      = make_shared_device(wgx);
                    auto d_wgy      = make_shared_device(wgy);
                    auto invocation = rocRoller::KernelInvocation{{totalSize, 1, 1}, {1, 1, 1}, 0};

                    kernel(invocation,
                           WGM,
                           numTilesM,
                           numTilesN,
                           evaluate(magicMultiple(rocRoller::Expression::literal(numTilesN))),
                           evaluate(magicShiftAndSign(rocRoller::Expression::literal(numTilesN))),
                           d_wgx.get(),
                           d_wgy.get());

                    CHECK_THAT(
                        hipMemcpy(
                            wgx.data(), d_wgx.get(), sizeof(int) * totalSize, hipMemcpyDefault),
                        HasHipSuccess(0));
                    CHECK_THAT(
                        hipMemcpy(
                            wgy.data(), d_wgy.get(), sizeof(int) * totalSize, hipMemcpyDefault),
                        HasHipSuccess(0));
                }

                // Build coverage
                std::map<std::pair<int, int>, int> coverage;
                for(uint i = 0; i < totalSize; ++i)
                {
                    auto mapped = std::pair<int, int>{wgx[i], wgy[i]};
                    coverage[mapped]++;
                }

                // Make sure everything is covered
                bool coverageOK = true;
                for(int i = 0; i < numTilesM; ++i)
                {
                    for(int j = 0; j < numTilesN; ++j)
                    {
                        auto mapped = std::pair<int, int>{i, j};
                        coverageOK &= coverage[mapped] == 1;
                    }
                }

                // Make sure first few entries are correct
                bool fastDimOK = true;
                int  blockSize = std::min(WGM, remapDim == 0 ? numTilesM : numTilesN);
                for(int i = 0; i < blockSize; ++i)
                {
                    auto value = remapDim == 0 ? wgx[i] : wgy[i];
                    if(value != i)
                        fastDimOK = false;
                }

                INFO(fmt::format(
                    " M {:2d} N {:2d} remap ({},{:2d})", numTilesM, numTilesN, remapDim, WGM));
                CHECK(coverage.size() == totalSize);
                CHECK(coverageOK);
                CHECK(fastDimOK);
            }
        }
    }
}
