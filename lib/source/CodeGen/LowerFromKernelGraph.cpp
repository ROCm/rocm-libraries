
#include "DataTypes/DataTypes.hpp"
#include "Utilities/Generator.hpp"
#include "Utilities/Utils.hpp"
#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /*
         * Code generation
         */
        struct NewCFCodeGeneratorVisitor
        {
            NewCFCodeGeneratorVisitor(KernelHypergraph                graph,
                                      std::shared_ptr<AssemblyKernel> kernel)
                : m_graph(graph)
                , m_kernel(kernel)
                , m_context(kernel->context())
                , m_fastArith{kernel->context()}
            {
            }

            Generator<Instruction> generate()
            {
                auto coords = CoordGraph::Transformer(
                    std::make_shared<CoordGraph::CoordinateHypergraph>(m_graph.coordinates),
                    m_context,
                    m_fastArith);

                co_yield Instruction::Comment("CFCodeGeneratorVisitor::generate() begin");
                co_yield setup();
                auto candidates = m_graph.control.roots().to<std::set>();
                co_yield generate(candidates, coords);
                co_yield Instruction::Comment("CFCodeGeneratorVisitor::generate() end");
            }

            inline Register::ValuePtr MkVGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Vector, type, count);
            }

            inline Register::ValuePtr MkSGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Scalar, type, count);
            }

            template <Graph::Direction dir>
            inline Register::ValuePtr getBufferSrd(int tag)
            {
                Register::ValuePtr srd;
                for(int const ntag : m_graph.coordinates.getNeighbours<dir>(tag))
                {
                    auto belem = m_graph.coordinates.get<CoordGraph::Buffer>(ntag);
                    if(belem)
                    {
                        return m_context->registerTagManager()->getRegister(ntag);
                    }
                }
                Throw<FatalError>("Buffer descriptor not found.");
            }

            template <Graph::Direction dir>
            std::pair<Register::ValuePtr, Register::ValuePtr> getOffsetAndStride(int tag)
            {
                Register::ValuePtr offset, stride;
                for(int const ntag : m_graph.coordinates.getNeighbours<dir>(tag))
                {
                    auto oelem = m_graph.coordinates.get<CoordGraph::Offset>(ntag);
                    if(oelem)
                    {
                        offset = m_context->registerTagManager()->getRegister(ntag);
                    }
                    auto selem = m_graph.coordinates.get<CoordGraph::Stride>(ntag);
                    if(selem)
                    {
                        stride = m_context->registerTagManager()->getRegister(ntag);
                    }
                }
                return {offset, stride};
            }

            inline ExpressionPtr L(auto const& x)
            {
                return Expression::literal(x);
            }

            inline Generator<Instruction> generate(auto& dest, ExpressionPtr expr) const
            {
                co_yield Expression::generate(dest, expr, m_context);
            }

            inline Generator<Instruction> copy(auto& dest, auto const& src) const
            {
                co_yield m_context->copier()->copy(dest, src);
            }

            Generator<Instruction> setup()
            {
                for(auto x : m_kernel->workgroupSize())
                    m_workgroupSize.push_back(x);
                for(auto x : m_kernel->workgroupIndex())
                    if(x)
                        m_workgroup.push_back(x->expression());
                for(auto x : m_kernel->workitemIndex())
                    if(x)
                        m_workitem.push_back(x->expression());

                co_return;
            }

            /**
             * Generate an index from `expr` and store in `dst`
             * register.  Destination register should be an Int64.
             */
            Generator<Instruction> generateOffset(Register::ValuePtr&       dst,
                                                  Expression::ExpressionPtr expr,
                                                  DataType                  dtype)
            {
                auto const& info     = DataTypeInfo::Get(dtype);
                auto        numBytes = Expression::literal(static_cast<uint>(info.elementSize));

                // TODO: Consider moving numBytes into input of this function.
                co_yield Expression::generate(dst, expr * numBytes, m_context);
            }

            bool hasGeneratedInputs(int const& tag)
            {
                auto inputTags
                    = m_graph.control.getInputNodeIndices<ControlHypergraph::Sequence>(tag);
                for(auto const& itag : inputTags)
                {
                    if(m_completedControlNodes.find(itag) == m_completedControlNodes.end())
                        return false;
                }
                return true;
            }

            /**
             * Generate code for the specified nodes and their standard (Sequence) dependencies.
             */
            Generator<Instruction> generate(std::set<int>           candidates,
                                            CoordGraph::Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    concatenate("KernelGraph::CFCodeGeneratorVisitor::generate: ", candidates));

                co_yield Instruction::Comment(concatenate("generate(", candidates, ")"));

                while(!candidates.empty())
                {
                    std::set<int> nodes;

                    // Find all candidate nodes whose inputs have been satisfied
                    for(auto const& tag : candidates)
                        if(hasGeneratedInputs(tag))
                            nodes.insert(tag);

                    // If there are none, we have a problem.
                    AssertFatal(!nodes.empty(),
                                "Invalid control graph!",
                                ShowValue(m_graph.control),
                                ShowValue(candidates));

                    // Generate code for all the nodes we found.

                    for(auto const& tag : nodes)
                    {
                        auto op = std::get<ControlHypergraph::Operation>(
                            m_graph.control.getElement(tag));
                        co_yield (*this)(tag, op, coords);
                    }

                    // Add output nodes to candidates.

                    for(auto const& tag : nodes)
                    {
                        auto outTags
                            = m_graph.control.getOutputNodeIndices<ControlHypergraph::Sequence>(
                                tag);
                        candidates.insert(outTags.begin(), outTags.end());
                    }

                    // Delete generated nodes from candidates.

                    for(auto const& node : nodes)
                        candidates.erase(node);
                }
            }

            template <Expression::CBinary Operation>
            Generator<Instruction> generateCommutativeBinaryOp(int dest, int a, int b)
            {
                auto lhs = m_context->registerTagManager()->getRegister(a);
                auto rhs = m_context->registerTagManager()->getRegister(b);

                auto const lhsInfo = DataTypeInfo::Get(lhs->variableType());
                auto const rhsInfo = DataTypeInfo::Get(rhs->variableType());

                AssertFatal(lhs->valueCount() * lhsInfo.packing
                                    == rhs->valueCount() * rhsInfo.packing
                                || lhs->valueCount() * lhsInfo.packing == 1
                                || rhs->valueCount() * rhsInfo.packing == 1,
                            "Commutative binary operation size mismatch.");

                auto regType    = Register::Type::Vector;
                auto varType    = VariableType::Promote(lhs->variableType(), rhs->variableType());
                auto valueCount = std::max(lhs->valueCount() * lhsInfo.packing,
                                           rhs->valueCount() * rhsInfo.packing);

                co_yield m_context->copier()->ensureType(lhs, lhs, regType);
                co_yield m_context->copier()->ensureType(rhs, rhs, regType);

                auto dst = m_context->registerTagManager()->getRegister(
                    dest, regType, varType, valueCount);
                co_yield Register::AllocateIfNeeded(dst);

                auto op = GetGenerator<Operation>(dst, lhs, rhs);

                if(lhsInfo.packing != rhsInfo.packing)
                {
                    // If the packing values of the datatypes are different, we need to
                    // convert the more packed value into the less packed value type.
                    // We can then perform the operation.
                    int packingRatio = std::max(lhsInfo.packing, rhsInfo.packing)
                                       / std::min(lhsInfo.packing, rhsInfo.packing);

                    if(rhsInfo.packing < lhsInfo.packing)
                        std::swap(lhs, rhs);

                    for(size_t i = 0; i < dst->valueCount(); i += packingRatio)
                    {
                        auto result = dst->element({i, i + packingRatio - 1});
                        co_yield generateConvertOp(
                            varType.dataType, result, rhs->element({i / packingRatio}));
                        for(size_t j = 0; j < packingRatio; j++)
                        {
                            auto lhsVal = lhs->valueCount() == 1 ? lhs : lhs->element({i + j});
                            co_yield op->generate(
                                result->element({j}), lhsVal, result->element({j}));
                        }
                    }
                }
                else
                {
                    for(size_t k = 0; k < valueCount; ++k)
                    {
                        auto lhsVal = lhs->valueCount() == 1 ? lhs : lhs->element({k});
                        auto rhsVal = rhs->valueCount() == 1 ? rhs : rhs->element({k});
                        co_yield op->generate(dst->element({k}), lhsVal, rhsVal);
                    }
                }
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::Operation const& operation,
                                              CoordGraph::Transformer const&      coords)
            {
                auto opName = toString(operation);
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::{}({})", opName, tag);
                co_yield Instruction::Comment(opName + " BEGIN");

                AssertFatal(m_completedControlNodes.find(tag) == m_completedControlNodes.end(),
                            ShowValue(operation));

                co_yield std::visit(*this,
                                    std::variant<int>(tag),
                                    operation,
                                    std::variant<CoordGraph::Transformer>(coords));

                co_yield Instruction::Comment(opName + " END");

                m_completedControlNodes.insert(tag);
            }

            Generator<Instruction> operator()(Operations::E_Neg, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Neg");

                auto src = m_context->registerTagManager()->getRegister(a);
                auto dst = m_context->registerTagManager()->getRegister(tag, src->placeholder());

                co_yield generateOp<Expression::Negate>(dst, src);
            }

            Generator<Instruction> operator()(Operations::E_Abs, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Abs");
                // TODO: Finish codegen for E_Abs
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Not, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Not");
                // TODO: Finish codegen for E_Not
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Add, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Add");

                co_yield generateCommutativeBinaryOp<Expression::Add>(tag, a, b);
            }

            Generator<Instruction> operator()(Operations::E_Sub, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Sub");

                auto lhs = m_context->registerTagManager()->getRegister(a);
                auto rhs = m_context->registerTagManager()->getRegister(b);
                auto dst = m_context->registerTagManager()->getRegister(tag, lhs->placeholder());

                co_yield generateOp<Expression::Subtract>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_Mul, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Mul");

                co_yield generateCommutativeBinaryOp<Expression::Multiply>(tag, a, b);
            }

            Generator<Instruction> operator()(Operations::E_Div, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Div");

                auto lhs = m_context->registerTagManager()->getRegister(a);
                auto rhs = m_context->registerTagManager()->getRegister(b);
                auto dst = m_context->registerTagManager()->getRegister(tag, lhs->placeholder());

                co_yield generateOp<Expression::Divide>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_And, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_And");
                // TODO: Finish codegen for E_And
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Or, int tag, int a, int b)
            {
                co_yield Instruction::Comment("GEN: E_Or");
                // TODO: Finish codegen for E_Or
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(int                          tag,
                                              ControlHypergraph::ElementOp eop,
                                              CoordGraph::Transformer      coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::ElementOp({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: ElementOp");

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Output of element op must be a single dimension.");

                int dest = connections[0].coordinate;
                co_yield std::visit(
                    [&](auto&& arg) -> Generator<Instruction> {
                        co_yield (*this)(arg, dest, eop.a, eop.b);
                    },
                    eop.op);
            }

            Generator<Instruction> operator()(int                              tag,
                                              ControlHypergraph::Kernel const& edge,
                                              CoordGraph::Transformer          coords)
            {
                co_yield Instruction::Comment("Begin Kernel");

                auto body = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                .to<std::set>();
                co_yield generate(body, coords);

                co_yield Instruction::Comment("End Kernel");
            }

            Generator<Instruction>
                operator()(int tag, ControlHypergraph::Scope const&, CoordGraph::Transformer coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Scope({})", tag);

                auto scope = std::make_shared<ScopeManager>(m_context);
                coords.setScope(scope);

                auto body = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                .to<std::set>();
                co_yield generate(body, coords);

                scope->release();
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::ForLoopOp const& op,
                                              CoordGraph::Transformer             coords)
            {
                // TODO: Logging level for these comments.
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::ForLoopOp({})",
                                                   tag);
                co_yield Instruction::Comment("For Loop Begin");

                auto topLabel = m_context->labelAllocator()->label("ForLoopTop");
                auto botLabel = m_context->labelAllocator()->label("ForLoopBottom");

                co_yield Instruction::Comment("Initialize For Loop");
                auto init = m_graph.control.getOutputNodeIndices<ControlHypergraph::Initialize>(tag)
                                .to<std::set>();
                co_yield generate(init, coords);

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1);
                auto loop_incr_tag = connections[0].coordinate;
                auto iterReg       = m_context->registerTagManager()->getRegister(loop_incr_tag);
                {
                    auto loopDims
                        = m_graph.coordinates.getOutputNodeIndices<CoordGraph::DataFlowEdge>(
                            loop_incr_tag);
                    for(auto const& dim : loopDims)
                    {
                        coords.setCoordinate(dim, iterReg->expression());
                    }
                }

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock For Loop");
                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(op.condition);
                auto conditionResult = conditionRegisterType == Register::Type::Special
                                               && conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();
                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(op.condition), m_context);
                co_yield m_context->brancher()->branchIfZero(
                    botLabel,
                    conditionResult,
                    concatenate("Condition: Top (jump to " + botLabel->toString() + " if false)"));

                co_yield Instruction::Label(topLabel);

                auto body = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                .to<std::set>();
                co_yield generate(body, coords);

                co_yield Instruction::Comment("For Loop Increment");
                auto incr
                    = m_graph.control.getOutputNodeIndices<ControlHypergraph::ForLoopIncrement>(tag)
                          .to<std::set>();
                co_yield generate(incr, coords);
                co_yield Instruction::Comment("Condition: Bottom (jump to " + topLabel->toString()
                                              + " if true)");

                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(op.condition), m_context);
                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    conditionResult,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                co_yield Instruction::Label(botLabel);

                co_yield Instruction::Comment("For Loop End");
                co_yield Instruction::Unlock("Unlock For Loop");
            }

            Generator<Instruction> operator()(int                                tag,
                                              ControlHypergraph::UnrollOp const& unroll,
                                              CoordGraph::Transformer            coords)
            {
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(int                              tag,
                                              ControlHypergraph::Assign const& assign,
                                              CoordGraph::Transformer          coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Assign({})", tag);
                auto varType = resultVariableType(assign.expression);

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid Assign operation; coordinate missing.");
                auto dim_tag = connections[0].coordinate;
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::Assign({}): {}", tag, dim_tag);

                auto dest = m_context->registerTagManager()->getRegister(
                    dim_tag, assign.regType, varType, assign.valueCount);

                auto scope = coords.getScope();
                if(scope && assign.localScope)
                {
                    scope->add(dim_tag);
                }

                co_yield Expression::generate(dest, assign.expression, m_context);
            }

            Generator<Instruction> operator()(int                                  tag,
                                              ControlHypergraph::Deallocate const& deallocate,
                                              CoordGraph::Transformer              coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Deallocate({})",
                                                   tag);
                auto dim_tag = m_graph.mapper.get<CoordGraph::Dimension>(tag);
                m_context->registerTagManager()->deleteRegister(dim_tag);
                co_return;
            }

            Generator<Instruction>
                operator()(int, ControlHypergraph::Barrier const&, CoordGraph::Transformer)
            {
                co_yield m_context->mem()->barrier();
            }

            Generator<Instruction> operator()(int                                    tag,
                                              ControlHypergraph::ComputeIndex const& ci,
                                              CoordGraph::Transformer                coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::ComputeIndex({}): {}/{}",
                    tag,
                    ci.offset,
                    ci.stride);
                co_yield Instruction::Comment(concatenate("GEN: ComputeIndex ", tag));

                auto scope    = coords.getScope();
                uint numBytes = DataTypeInfo::Get(ci.valueType).elementSize;

                coords.setCoordinate(ci.increment, L(0u));
                for(auto const tag_ : ci.zero)
                    coords.setCoordinate(tag_, L(0u));

                auto offsetReg = m_context->registerTagManager()->getRegister(
                    ci.offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("offset", tag));
                co_yield Register::AllocateIfNeeded(offsetReg);
                scope->add(ci.offset);

                if(ci.base < 0)
                {
                    auto indexExpr = ci.forward ? coords.forward({ci.target})[0]
                                                : coords.reverse({ci.target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Offset({}): {}", ci.offset, toString(indexExpr));
                    co_yield generate(offsetReg, indexExpr * L(numBytes));
                }

                auto strideReg = m_context->registerTagManager()->getRegister(
                    ci.stride, Register::Type::Scalar, ci.strideType, 1);
                strideReg->setName(concatenate("stride", tag));
                co_yield Register::AllocateIfNeeded(strideReg);
                scope->add(ci.stride);

                if(ci.stride > 0)
                {
                    auto indexExpr = ci.forward
                                         ? coords.forwardStride(ci.increment, L(1), {ci.target})[0]
                                         : coords.reverseStride(ci.increment, L(1), {ci.target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Stride({}): {}", ci.stride, toString(indexExpr));
                    co_yield generate(strideReg, indexExpr * L(numBytes));
                }

                auto         user = m_graph.coordinates.get<CoordGraph::User>(ci.target);
                VariableType bufferPointer{DataType::None, PointerType::Buffer};
                auto         bufferReg = m_context->registerTagManager()->getRegister(
                    ci.buffer, Register::Type::Scalar, bufferPointer, 1);
                bufferReg->setName(concatenate("buffer", tag));
                co_yield Register::AllocateIfNeeded(bufferReg);
                auto basePointer = MkSGPR(DataType::Int64);
                auto bufDesc     = BufferDescriptor(bufferReg, m_context);
                co_yield m_context->argLoader()->getValue(user->argumentName(), basePointer);
                co_yield bufDesc.setBasePointer(basePointer);
                co_yield bufDesc.setDefaultOpts();
                scope->add(ci.buffer);
            }

            Generator<Instruction> operator()(int                                     tag,
                                              ControlHypergraph::SetCoordinate const& setCoordinate,
                                              CoordGraph::Transformer                 coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::SetCoordinate({}): {}",
                    tag,
                    Expression::toString(setCoordinate.value));

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid SetCoordinate operation; coordinate missing.");
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto body = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                .to<std::set>();
                co_yield generate(body, coords);
            }

            Generator<Instruction> operator()(int                                  tag,
                                              ControlHypergraph::LoadLinear const& edge,
                                              CoordGraph::Transformer              coords)
            {
                Throw<FatalError>("LoadLinear present in kernel graph.");
            }

            Generator<Instruction> loadMacroTileVGPR(int                                 tag,
                                                     ControlHypergraph::LoadTiled const& load,
                                                     CoordGraph::Transformer             coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileVGPR()");
                co_yield Instruction::Comment("GEN: loadMacroTileVGPR");

                auto [user_tag, user]         = m_graph.getDimension<CoordGraph::User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto bufOpt = BufferInstructionOptions();

                auto i_thr_x = m_graph.mapper.get<CoordGraph::ThreadTileIndex>(tag, 0);
                auto i_thr_y = m_graph.mapper.get<CoordGraph::ThreadTileIndex>(tag, 1);

                auto [row_offset_reg, row_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(i_thr_x);
                auto [col_offset_reg, col_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(i_thr_y);
                auto bufferSrd = getBufferSrd<Graph::Direction::Upstream>(i_thr_x);
                auto bufDesc   = BufferDescriptor(bufferSrd, m_context);

                auto tmpl = MkVGPR(load.vtype, product(mac_tile.subTileSizes));
                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                // TODO: multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }

                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> loadMacroTileWAVECI(int                                 tag,
                                                       ControlHypergraph::LoadTiled const& load,
                                                       CoordGraph::Transformer             coords,
                                                       int                                 sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECI({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECI");

                auto [user_tag, user]           = m_graph.getDimension<CoordGraph::User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<CoordGraph::WaveTile>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<CoordGraph::MacroTile>(tag);
                auto [vgpr_tag, vgpr]           = m_graph.getDimension<CoordGraph::VGPR>(tag);

                Register::ValuePtr basePointer;
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto n_wave_tag  = m_graph.mapper.get<CoordGraph::WaveTileNumber>(tag, sdim);
                auto mac_num_tag = m_graph.mapper.get<CoordGraph::MacroTileNumber>(tag, sdim);

                auto [wave_offset_reg, wave_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(n_wave_tag);
                auto [vgpr_offset_reg, vgpr_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(vgpr_tag);
                auto bufferSrd = getBufferSrd<Graph::Direction::Upstream>(mac_num_tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(wave_offset_reg, "Invalid WAVE offset register.");
                AssertFatal(vgpr_offset_reg, "Invalid VGPR offset register.");
                AssertFatal(vgpr_stride_reg, "Invalid VGPR stride register.");

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; a += 2)
                    {
                        co_yield copy(offset1, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());
                        co_yield copy(offset2, vgpr_offset_reg);
                        co_yield generate(vgpr_offset_reg,
                                          vgpr_offset_reg->expression()
                                              + vgpr_stride_reg->expression());

                        co_yield m_context->mem()->loadAndPackBuffer(
                            vgpr->element({static_cast<int>(a / 2)}),
                            offset1,
                            offset2,
                            bufDesc,
                            bufOpt);
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    co_yield copy(vgpr_offset_reg, wave_offset_reg);

                    for(uint a = 0; a < num_vgpr; ++a)
                    {
                        co_yield m_context->mem()->loadBuffer(vgpr->element({static_cast<int>(a)}),
                                                              vgpr_offset_reg->subset({0}),
                                                              0,
                                                              bufDesc,
                                                              bufOpt,
                                                              numBytes);

                        if(a < num_vgpr - 1)
                            co_yield generate(vgpr_offset_reg,
                                              vgpr_offset_reg->expression()
                                                  + vgpr_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> loadMacroTileWAVECIACCUM(
                int tag, ControlHypergraph::LoadTiled const& load, CoordGraph::Transformer coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECIACCUM({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECIACCUM");

                auto [user_tag, user]           = m_graph.getDimension<CoordGraph::User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<CoordGraph::WaveTile>(tag);
                auto mac_tile_tag               = m_graph.mapper.get<CoordGraph::MacroTile>(tag);
                auto vgpr_tag                   = m_graph.mapper.get<CoordGraph::VGPR>(tag);
                auto vgpr_block_tag = m_graph.mapper.get<CoordGraph::VGPRBlockNumber>(tag);
                auto vgpr_index_tag = m_graph.mapper.get<CoordGraph::VGPRBlockIndex>(tag);

                // Move the argument pointer into v_ptr
                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto [vgpr_block_offset_reg, vgpr_block_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(vgpr_block_tag);
                auto [vgpr_index_offset_reg, vgpr_index_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(vgpr_index_tag);
                auto bufferSrd = getBufferSrd<Graph::Direction::Upstream>(vgpr_block_tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(vgpr_block_offset_reg, "Invalid VGPR BLOCK offset register.");
                AssertFatal(vgpr_block_stride_reg, "Invalid VGPR BLOCK stride register.");
                AssertFatal(vgpr_index_stride_reg, "Invalid VGPR INDEX stride register.");

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                    {
                        co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                        for(uint aidx = 0; aidx < 4; aidx += 2)
                        {
                            uint a = ablk * 4 + aidx;

                            co_yield copy(offset1, vgpr_index_offset_reg);
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());
                            co_yield copy(offset2, vgpr_index_offset_reg);
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());

                            co_yield m_context->mem()->loadAndPackBuffer(
                                vgpr->element({static_cast<int>(a / 2)}),
                                offset1,
                                offset2,
                                bufDesc,
                                bufOpt);
                        }
                        if(ablk < num_vgpr / 4 - 1)
                            co_yield generate(vgpr_block_offset_reg,
                                              vgpr_block_offset_reg->expression()
                                                  + vgpr_block_stride_reg->expression());
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                    {
                        co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                        for(uint aidx = 0; aidx < 4; ++aidx)
                        {
                            uint a = ablk * 4 + aidx;

                            co_yield m_context->mem()->loadBuffer(
                                vgpr->element({static_cast<int>(a)}),
                                vgpr_index_offset_reg->subset({0}),
                                0,
                                bufDesc,
                                bufOpt,
                                numBytes);

                            if(aidx < 3)
                                co_yield generate(vgpr_index_offset_reg,
                                                  vgpr_index_offset_reg->expression()
                                                      + vgpr_index_stride_reg->expression());
                        }
                        if(ablk < num_vgpr / 4 - 1)
                            co_yield generate(vgpr_block_offset_reg,
                                              vgpr_block_offset_reg->expression()
                                                  + vgpr_block_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::LoadTiled const& load,
                                              CoordGraph::Transformer             coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadTiled({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: LoadTiled");

                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield loadMacroTileVGPR(tag, load, coords);
                    break;
                case MemoryType::WAVE:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileWAVECI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileWAVECI(tag, load, coords, 0);
                        break;
                    case LayoutType::MATRIX_ACCUMULATOR:
                        co_yield loadMacroTileWAVECIACCUM(tag, load, coords);
                        break;
                    default:
                        Throw<FatalError>("Layout type not supported yet.");
                    }
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction> operator()(int                                   tag,
                                              ControlHypergraph::LoadLDSTile const& load,
                                              CoordGraph::Transformer               coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadLDSTile({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: LoadLDSTile");

                auto [user_tag, user] = m_graph.getDimension<CoordGraph::User>(tag);
                auto [lds_tag, lds]   = m_graph.getDimension<CoordGraph::LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);

                auto vtype = ldsAllocation->variableType();

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({lds_tag})[0])
                      * Expression::literal(product(tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                co_yield generate(lds_offset,
                                  coords.getTransducer()(lds_offset_expr * L(numBytes)));

                auto vgpr = m_context->registerTagManager()->getRegister(tile_tag);

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield m_context->mem()->loadLocal(
                        vgpr->element(Generated(iota(i * n, i * n + n))),
                        lds_offset,
                        i * n * numBytes,
                        n * numBytes);
                }
            }

            Generator<Instruction> operator()(int                                tag,
                                              ControlHypergraph::LoadVGPR const& load,
                                              CoordGraph::Transformer            coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadVGPR({})", tag);
                co_yield Instruction::Comment("GEN: LoadVGPR");

                auto [userTag, user] = m_graph.getDimension<CoordGraph::User>(tag);
                auto [vgprTag, vgpr] = m_graph.getDimension<CoordGraph::VGPR>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR({}): User({}), VGPR({})",
                    tag,
                    userTag,
                    vgprTag);

                auto dst = m_context->registerTagManager()->getRegister(
                    vgprTag, Register::Type::Vector, load.vtype.dataType);
                co_yield Register::AllocateIfNeeded(dst);

                if(load.scalar)
                {
                    if(load.vtype.isPointer())
                        co_yield loadVGPRFromScalarPointer(user, dst, coords);
                    else
                        co_yield loadVGPRFromScalarValue(user, dst, coords);
                }
                else
                {
                    co_yield loadVGPRFromGlobalArray(userTag, user, dst, coords);
                }
            }

            Generator<Instruction> loadVGPRFromScalarValue(CoordGraph::User                 user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           CoordGraph::Transformer          coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar value");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar value");

                Register::ValuePtr s_value;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_value);
                co_yield m_context->copier()->copy(vgpr, s_value, "Move value");
            }

            Generator<Instruction> loadVGPRFromScalarPointer(CoordGraph::User                 user,
                                                             std::shared_ptr<Register::Value> vgpr,
                                                             CoordGraph::Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar pointer");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar pointer");

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, v_ptr, nullptr, numBytes);
            }

            Generator<Instruction> loadVGPRFromGlobalArray(int                              userTag,
                                                           CoordGraph::User                 user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           CoordGraph::Transformer          coords)
            {
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                co_yield Instruction::Comment("GEN: LoadVGPR; user index");

                auto indexes = coords.reverse({userTag});
                co_yield generateOffset(offset, indexes[0], vgpr->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, v_ptr, offset, numBytes);
            }

            Generator<Instruction> operator()(int                                tag,
                                              ControlHypergraph::Multiply const& mult,
                                              CoordGraph::Transformer            coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Multiply({})", tag);
                co_yield Instruction::Comment("GEN: Multiply");

                auto [userA_tag, _uA] = m_graph.getDimension<CoordGraph::User>(tag, 0);
                auto [userB_tag, _uB] = m_graph.getDimension<CoordGraph::User>(tag, 1);

                auto [waveA_tag, waveA] = m_graph.getDimension<CoordGraph::WaveTile>(tag, 0);
                auto [waveB_tag, waveB] = m_graph.getDimension<CoordGraph::WaveTile>(tag, 1);

                auto [macA_tag, macA] = m_graph.getDimension<CoordGraph::MacroTile>(tag, 0);
                auto [macB_tag, macB] = m_graph.getDimension<CoordGraph::MacroTile>(tag, 1);

                auto n_macA_y_tags
                    = m_graph.coordinates
                          .findNodes(userA_tag,
                                     [&](int index) -> bool {
                                         auto node
                                             = m_graph.coordinates.get<CoordGraph::MacroTileNumber>(
                                                 index);
                                         if(node)
                                             return node->dim == 1;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_macA_y_tags.size() == 1);

                auto n_waveA_y_tags
                    = m_graph.coordinates
                          .findNodes(userA_tag,
                                     [&](int index) -> bool {
                                         auto node
                                             = m_graph.coordinates.get<CoordGraph::WaveTileNumber>(
                                                 index);
                                         if(node)
                                             return node->dim == 1;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveA_y_tags.size() == 1);

                auto n_macB_x_tags
                    = m_graph.coordinates
                          .findNodes(userB_tag,
                                     [&](int index) -> bool {
                                         auto node
                                             = m_graph.coordinates.get<CoordGraph::MacroTileNumber>(
                                                 index);
                                         if(node)
                                             return node->dim == 0;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_macA_y_tags.size() == 1);

                auto n_waveB_x_tags
                    = m_graph.coordinates
                          .findNodes(userB_tag,
                                     [&](int index) -> bool {
                                         auto node
                                             = m_graph.coordinates.get<CoordGraph::WaveTileNumber>(
                                                 index);
                                         if(node)
                                             return node->dim == 0;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveB_x_tags.size() == 1);

                auto n_waveA_y
                    = *m_graph.coordinates.get<CoordGraph::WaveTileNumber>(n_waveA_y_tags.front());
                auto n_waveB_x
                    = *m_graph.coordinates.get<CoordGraph::WaveTileNumber>(n_waveB_x_tags.front());

                auto loadAB = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                  .to<std::set>();

                auto [mac_offset_x_reg, mac_stride_x_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(n_macA_y_tags.front());
                auto [wave_offset_x_reg, wave_stride_x_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(n_waveA_y_tags.front());

                auto [mac_offset_y_reg, mac_stride_y_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(n_macB_x_tags.front());
                auto [wave_offset_y_reg, wave_stride_y_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(n_waveB_x_tags.front());

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto [D_tag, _D] = m_graph.getDimension<CoordGraph::MacroTile>(tag, 2);

                auto D = m_context->registerTagManager()->getRegister(
                    D_tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto completed = m_completedControlNodes;

                // D is not initialized here

                co_yield copy(wave_offset_x_reg, mac_offset_x_reg);
                co_yield copy(wave_offset_y_reg, mac_offset_y_reg);

                uint const num_wave_tiles = macA.sizes[1] / waveA.sizes[1];
                for(uint k = 0; k < num_wave_tiles; k++)
                {
                    m_completedControlNodes = completed; // TODO: remove this?

                    // A WaveTile number; tall-skinny column block
                    coords.setCoordinate(n_waveA_y_tags.front(), literal(k));
                    // B WaveTile number; short-fat row block
                    coords.setCoordinate(n_waveB_x_tags.front(), literal(k));

                    co_yield generate(loadAB, coords);

                    waveA.vgpr = m_context->registerTagManager()->getRegister(macA_tag);
                    waveB.vgpr = m_context->registerTagManager()->getRegister(macB_tag);

                    Expression::ExpressionPtr A = std::make_shared<Expression::Expression>(
                        std::make_shared<CoordGraph::WaveTile>(waveA));
                    Expression::ExpressionPtr B = std::make_shared<Expression::Expression>(
                        std::make_shared<CoordGraph::WaveTile>(waveB));

                    co_yield generate(D,
                                      std::make_shared<Expression::Expression>(
                                          Expression::MatrixMultiply(A, B, D->expression())));

                    co_yield generate(wave_offset_x_reg,
                                      wave_offset_x_reg->expression()
                                          + wave_stride_x_reg->expression());

                    co_yield generate(wave_offset_y_reg,
                                      wave_offset_y_reg->expression()
                                          + wave_stride_y_reg->expression());
                }
            }

            Generator<Instruction> operator()(int                                         tag,
                                              ControlHypergraph::TensorContraction const& mul,
                                              CoordGraph::Transformer                     coords)
            {
                Throw<FatalError>("TensorContraction present in kernel graph.");
            }

            Generator<Instruction> operator()(int                                   tag,
                                              ControlHypergraph::StoreLinear const& edge,
                                              CoordGraph::Transformer               coords)
            {
                Throw<FatalError>("StoreLinear present in kernel graph.");
            }

            Generator<Instruction> storeMacroTileVGPR(int                                  tag,
                                                      ControlHypergraph::StoreTiled const& store,
                                                      CoordGraph::Transformer              coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileVGPR()");
                co_yield Instruction::Comment("GEN: storeMacroTileVGPR");

                auto [user_tag, user]         = m_graph.getDimension<CoordGraph::User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                auto i_thr_x = m_graph.mapper.get<CoordGraph::ThreadTileIndex>(tag, 0);
                auto i_thr_y = m_graph.mapper.get<CoordGraph::ThreadTileIndex>(tag, 1);

                auto [row_offset_reg, row_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(i_thr_x);
                auto [col_offset_reg, col_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Upstream>(i_thr_y);

                auto bufferSrd = getBufferSrd<Graph::Direction::Upstream>(i_thr_x);
                auto bufDesc   = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt    = BufferInstructionOptions();

                // TODO multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->storeBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            col_offset_reg->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                        }
                    }
                    if(i < m - 1)
                    {
                        co_yield generate(row_offset_reg,
                                          row_offset_reg->expression()
                                              + row_stride_reg->expression());
                    }
                }
            }

            Generator<Instruction> storeMacroTileWAVECI(int                                  tag,
                                                        ControlHypergraph::StoreTiled const& store,
                                                        CoordGraph::Transformer              coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVE()");
                co_yield Instruction::Comment("GEN: storeMacroTileWAVE");

                auto [user_tag, user]           = m_graph.getDimension<CoordGraph::User>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<CoordGraph::MacroTile>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<CoordGraph::WaveTile>(tag);
                auto [vgpr_tag, vgpr]           = m_graph.getDimension<CoordGraph::VGPR>(tag);
                auto vgpr_block_tag = m_graph.mapper.get<CoordGraph::VGPRBlockNumber>(tag);
                auto vgpr_index_tag = m_graph.mapper.get<CoordGraph::VGPRBlockIndex>(tag);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto agpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                AssertFatal(agpr->registerCount() == num_vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto [vgpr_block_offset_reg, vgpr_block_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Downstream>(vgpr_block_tag);
                auto [vgpr_index_offset_reg, vgpr_index_stride_reg]
                    = getOffsetAndStride<Graph::Direction::Downstream>(vgpr_index_tag);
                auto bufferSrd = getBufferSrd<Graph::Direction::Downstream>(vgpr_block_tag);

                auto bufDesc = BufferDescriptor(bufferSrd, m_context);
                auto bufOpt  = BufferInstructionOptions();

                AssertFatal(vgpr_block_offset_reg, "Invalid VGPR BLOCK offset register.");
                AssertFatal(vgpr_block_stride_reg, "Invalid VGPR BLOCK stride register.");
                AssertFatal(vgpr_index_stride_reg, "Invalid VGPR INDEX stride register.");

                auto numBytes  = DataTypeInfo::Get(store.dataType).elementSize;
                auto value     = MkVGPR(agpr->variableType());
                auto converted = MkVGPR(store.dataType);

                for(uint ablk = 0; ablk < num_vgpr / 4; ++ablk)
                {
                    co_yield copy(vgpr_index_offset_reg, vgpr_block_offset_reg);
                    for(uint aidx = 0; aidx < 4; ++aidx)
                    {
                        uint a = ablk * 4 + aidx;
                        if(value->variableType() != store.dataType)
                        {
                            co_yield m_context->copier()->copy(
                                value, agpr->element({static_cast<int>(a)}));
                            co_yield Expression::generate(
                                converted,
                                convert(store.dataType,
                                        std::make_shared<Expression::Expression>(value)),
                                m_context);
                        }
                        else
                        {
                            co_yield m_context->copier()->copy(
                                converted, agpr->element({static_cast<int>(a)}));
                        }

                        co_yield m_context->mem()->storeBuffer(converted,
                                                               vgpr_index_offset_reg->subset({0}),
                                                               0,
                                                               bufDesc,
                                                               bufOpt,
                                                               numBytes);

                        if(aidx < 3)
                            co_yield generate(vgpr_index_offset_reg,
                                              vgpr_index_offset_reg->expression()
                                                  + vgpr_index_stride_reg->expression());
                    }
                    if(ablk < num_vgpr / 4 - 1)
                        co_yield generate(vgpr_block_offset_reg,
                                          vgpr_block_offset_reg->expression()
                                              + vgpr_block_stride_reg->expression());
                }
            }

            Generator<Instruction> operator()(int                                  tag,
                                              ControlHypergraph::StoreTiled const& store,
                                              CoordGraph::Transformer              coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreTiled()");
                co_yield Instruction::Comment("GEN: StoreTiled");

                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(tag, store, coords);
                    break;
                case MemoryType::WAVE:
                    co_yield storeMacroTileWAVECI(tag, store, coords);
                    break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction> operator()(int                                    tag,
                                              ControlHypergraph::StoreLDSTile const& store,
                                              CoordGraph::Transformer                coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreLDSTiled({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: StoreLDSTile");

                auto [lds_tag, lds]   = m_graph.getDimension<CoordGraph::LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                // Temporary register that is used to copy the data from global memory to
                // local memory.
                auto vgpr  = m_context->registerTagManager()->getRegister(tile_tag);
                auto vtype = vgpr->variableType();

                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                auto ldsAllocation = Register::Value::AllocateLDS(
                    m_context, vtype, product(tile.subTileSizes) * product(m_workgroupSize));

                m_context->registerTagManager()->addRegister(lds_tag, ldsAllocation);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({lds_tag})[0])
                      * Expression::literal(product(tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;
                co_yield generate(lds_offset,
                                  coords.getTransducer()(lds_offset_expr * L(numBytes)));

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    // Store the value into local memory
                    co_yield m_context->mem()->storeLocal(
                        lds_offset->subset({0}),
                        vgpr->element(Generated(iota(i * n, i * n + n))),
                        i * n * numBytes,
                        n * numBytes);
                }
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::StoreVGPR const& store,
                                              CoordGraph::Transformer             coords)
            {
                co_yield Instruction::Comment("GEN: StoreVGPR");

                auto [vgprTag, vgpr] = m_graph.getDimension<CoordGraph::VGPR>(tag);
                auto [userTag, user] = m_graph.getDimension<CoordGraph::User>(tag);

                auto src = m_context->registerTagManager()->getRegister(vgprTag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto indexes = coords.forward({userTag});

                co_yield Instruction::Comment("GEN: StoreVGPR; user index");
                co_yield offset->allocate();
                co_yield generateOffset(offset, indexes[0], src->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, src->variableType().getPointer(), 1);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(src->variableType()).elementSize;
                co_yield m_context->mem()->store(
                    MemoryInstructions::MemoryKind::Flat, v_ptr, src, offset, numBytes);
            }

        private:
            KernelHypergraph                m_graph;
            std::shared_ptr<Context>        m_context;
            std::shared_ptr<AssemblyKernel> m_kernel;

            std::set<int> m_completedControlNodes;

            std::vector<ExpressionPtr> m_workgroup;
            std::vector<ExpressionPtr> m_workitem;
            std::vector<unsigned int>  m_workgroupSize;
            FastArithmetic             m_fastArith;
        };

        Generator<Instruction> generate(KernelHypergraph                graph,
                                        std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::generate");
            rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}",
                                               graph.toDOT(true));

            auto visitor = NewCFCodeGeneratorVisitor(graph, kernel);

            co_yield visitor.generate();
        }
    }
}
