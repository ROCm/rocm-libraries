// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CodeGen/Arithmetic/BitFieldExtract.hpp>
#include <rocRoller/CodeGen/ExchangeGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        ExchangeGenerator::ExchangeGenerator(KernelGraphPtr graph, ContextPtr context)
            : m_graph(graph)
            , m_context(context)
            , m_fastArith(FastArithmetic(context))
        {
        }

        Generator<Instruction>
            ExchangeGenerator::genExchange(int tag, Exchange const& exchange, Transformer coords)
        {
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto oMacTileTag             = m_graph->mapper.get(tag, NaryArgument::DEST);

            uint waveMN, waveK, miMN, miK;
            switch(macTile.layoutType)
            {
            case LayoutType::MATRIX_A:
                waveMN = waveTile.sizes[0];
                waveK  = waveTile.sizes[1];
                miMN   = macTile.miTileSizes[0];
                miK    = macTile.miTileSizes[1];
                break;
            case LayoutType::MATRIX_B:
                waveMN = waveTile.sizes[1];
                waveK  = waveTile.sizes[0];
                miMN   = macTile.miTileSizes[1];
                miK    = macTile.miTileSizes[0];
                break;
            default:
                Throw<FatalError>("Layout type not supported yet for Exchange.");
            }

            const uint waveTileSize = waveTile.sizes[0] * waveTile.sizes[1];

            Expression::ExpressionPtr waveTileExpr, expectedExpr;

            auto setCoordinates = [&](int tag) {
                auto [required, path]
                    = findRequiredCoordinates(tag, Graph::Direction::Downstream, *m_graph);

                auto tmpCoords = coords;
                for(auto r : required)
                {
                    auto expr = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{r, Register::Type::Vector, DataType::UInt32});
                    tmpCoords.setCoordinate(r, expr);
                }
                return tmpCoords;
            };

            {
                auto tmpCoords = setCoordinates(waveTileTag);
                waveTileExpr   = tmpCoords.reverse({waveTileTag})[0];
                AssertFatal(waveTileExpr != nullptr, "waveTile expr is null");
            }

            if(waveMN == 64)
            {
                Expression::ExpressionPtr vgprIndexExpr, simdIndexBlockExpr;
                auto [vgprIndexTag, vgprIndex] = m_graph->getDimension<VGPRBlockIndex>(tag);
                auto [simdIndexBlockTag, simdIndexBlock] = m_graph->getDimension<Adhoc>(tag, 2);

                {
                    auto tmpCoords = setCoordinates(vgprIndexTag);
                    vgprIndexExpr  = tmpCoords.reverse({vgprIndexTag})[0];
                    AssertFatal(vgprIndexExpr != nullptr, "vgprIndex expr is null");
                    expectedExpr
                        = (waveTileExpr / (Expression::literal(waveTileSize) / vgprIndex.size));
                    AssertFatal(Expression::identical(m_fastArith(vgprIndexExpr),
                                                      m_fastArith(expectedExpr)),
                                "Exchange: VGPRIndex must be the slowest running dimension",
                                ShowValue(m_fastArith(vgprIndexExpr)),
                                ShowValue(m_fastArith(expectedExpr)));
                }

                {
                    auto tmpCoords     = setCoordinates(simdIndexBlockTag);
                    simdIndexBlockExpr = tmpCoords.reverse({simdIndexBlockTag})[0];
                    AssertFatal(simdIndexBlockExpr != nullptr, "simdIndexBlock expr is null");
                    expectedExpr = waveTileExpr % simdIndexBlock.size;
                    AssertFatal(Expression::identical(m_fastArith(simdIndexBlockExpr),
                                                      m_fastArith(expectedExpr)),
                                "Exchange: SIMDIndexBlock must be the fastest running dimension");
                }
            }

            const uint wfs = m_context->kernel()->wavefront_size();
            // Exchange tile fixed size: 256 = 64 x 4 or 32 x 8
            const uint numVgpr = 256 / wfs;

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto packedVariableType = DataTypeInfo::Get(exchange.varType).packedVariableType();

            if(packedVariableType
               && m_context->kernelOptions()->scaleSkipPermlane
                      == rocRoller::ScaleSkipPermlaneMode::None)
            {
                auto               allocOptions = Register::AllocationOptions::FullyContiguous();
                Register::ValuePtr temp;
                if(m_context->registerTagManager()->hasRegister(oMacTileTag))
                {
                    temp = m_context->registerTagManager()->getRegister(oMacTileTag);
                }
                else
                {
                    temp = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, exchange.varType, numVgpr, allocOptions);
                }
                for(auto index = 0; index < numVgpr; index++)
                    co_yield generateOp<Expression::BitFieldExtract>(
                        temp->element({index}),
                        vgpr,
                        Expression::BitFieldExtract{
                            {}, exchange.varType.dataType, static_cast<uint32_t>(index * 8), 8});
                vgpr = temp;
            }

            if(m_context->kernelOptions()->scaleSkipPermlane
               != rocRoller::ScaleSkipPermlaneMode::None)
            {
                AssertFatal(m_context->registerTagManager()->hasRegister(oMacTileTag),
                            ShowValue(oMacTileTag));
            }
            else
            {
                AssertFatal(vgpr->registerCount() == numVgpr);

                if(!m_context->registerTagManager()->hasRegister(oMacTileTag))
                {
                    m_context->registerTagManager()->addRegister(oMacTileTag, vgpr);
                }

                if((waveMN == 64 && miMN == 16) || (waveMN == 32 && miMN == 16))
                {
                    for(uint32_t i = 0; i < numVgpr; i += 2)
                    {
                        co_yield_(Instruction::InoutInstruction(
                            "v_permlane16_swap_b32",
                            {vgpr->element({i}), vgpr->element({i + 1})},
                            {},
                            ""));
                    }
                    for(uint32_t i = 0; i < numVgpr / 2; i++)
                    {
                        co_yield_(Instruction::InoutInstruction(
                            "v_permlane32_swap_b32",
                            {vgpr->element({i}), vgpr->element({i + 2})},
                            {},
                            ""));
                    }
                }
                else if(waveMN == 64 && miMN == 32)
                {
                    for(uint32_t i = 0; i < numVgpr; i += 2)
                    {
                        co_yield_(Instruction::InoutInstruction(
                            "v_permlane32_swap_b32",
                            {vgpr->element({i}), vgpr->element({i + 1})},
                            {},
                            ""));
                    }
                }
                else if(waveMN == 32 && miMN == 32)
                {
                    for(uint32_t i = 0; i < numVgpr / 2; i++)
                    {
                        co_yield_(Instruction::InoutInstruction(
                            "v_permlane32_swap_b32",
                            {vgpr->element({i}), vgpr->element({i + 2})},
                            {},
                            ""));
                    }
                    for(uint32_t i = 0; i < numVgpr; i += 2)
                    {
                        co_yield_(Instruction::InoutInstruction(
                            "v_permlane32_swap_b32",
                            {vgpr->element({i}), vgpr->element({i + 1})},
                            {},
                            ""));
                    }
                }
                else
                {
                    Throw<FatalError>("Exchange for the given tile specification not supported.",
                                      ShowValue(waveTile.sizes[0]),
                                      ShowValue(macTile.miTileSizes[0]));
                }
            }
        }

    }
}
