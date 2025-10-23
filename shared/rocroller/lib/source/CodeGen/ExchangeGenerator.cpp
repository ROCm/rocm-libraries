
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
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;
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
            auto [waveTileTag, waveTile]   = m_graph->getDimension<WaveTile>(tag);
            auto [macTileTag, macTile]     = m_graph->getDimension<MacroTile>(tag);
            auto [vgprTag, vgprDim]        = m_graph->getDimension<VGPR>(tag);
            auto [simdIndexTag, simdIndex] = m_graph->getDimension<Adhoc>(tag, 1);
            auto oMacTileTag               = m_graph->mapper.get(tag, NaryArgument::DEST);

            const uint waveTileSize = waveTile.sizes[0] * waveTile.sizes[1];

            Expression::ExpressionPtr waveTileExpr, simdIndexExpr, vgprExpr, expectedExpr;

            {
                auto [required, path]
                    = findRequiredCoordinates(waveTileTag, Graph::Direction::Downstream, *m_graph);

                for(auto r : required)
                {
                    auto expr = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{r, Register::Type::Vector, DataType::UInt32});
                    coords.setCoordinate(r, expr);
                }

                waveTileExpr = coords.reverse({waveTileTag})[0];
            }

            {
                auto [required, path]
                    = findRequiredCoordinates(vgprTag, Graph::Direction::Downstream, *m_graph);

                for(auto r : required)
                {
                    if(r == vgprTag)
                        continue;
                    auto expr = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{r, Register::Type::Vector, DataType::UInt32});
                    coords.setCoordinate(r, expr);
                }

                vgprExpr     = coords.reverse({vgprTag})[0];
                expectedExpr = (waveTileExpr / (Expression::literal(waveTileSize) / vgprDim.size));
                AssertFatal(Expression::identical(m_fastArith(vgprExpr), m_fastArith(expectedExpr)),
                            "Exchange: VGPR must be the slowest running dimension",
                            ShowValue(m_fastArith(vgprExpr)),
                            ShowValue(m_fastArith(expectedExpr)));
            }

            {
                auto [required, path]
                    = findRequiredCoordinates(simdIndexTag, Graph::Direction::Downstream, *m_graph);

                for(auto r : required)
                {
                    auto expr = std::make_shared<Expression::Expression>(
                        Expression::DataFlowTag{r, Register::Type::Vector, DataType::UInt32});
                    coords.setCoordinate(r, expr);
                }

                simdIndexExpr = coords.reverse({simdIndexTag})[0];
                expectedExpr  = waveTileExpr % simdIndex.size;
                AssertFatal(
                    Expression::identical(m_fastArith(simdIndexExpr), m_fastArith(expectedExpr)),
                    "Exchange: SIMDIndex must be the fastest running dimension");
            }

            const uint wfs = m_context->kernel()->wavefront_size();
            // Exchange tile fixed size: 64 x 4
            const uint numVgpr = 64 * 4 / wfs;

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto packedVariableType = DataTypeInfo::Get(exchange.varType).packedVariableType();

            if(packedVariableType && !m_context->kernelOptions()->scaleSkipPermlane)
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
                        Expression::BitFieldExtract{{}, exchange.varType.dataType, index * 8, 8});
                vgpr = temp;
            }

            if(m_context->kernelOptions()->scaleSkipPermlane)
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

                if(Expression::identical(vgprDim.size, Expression::literal(4u)))
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
                else if(Expression::identical(vgprDim.size, Expression::literal(2u)))
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
                else
                {
                    Throw<FatalError>("Exchange for the given vgpr size not supported.");
                }
            }
        }

    }
}
