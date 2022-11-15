
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
#include <rocRoller/KernelGraph/CoordinateTransform/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        using namespace CoordinateTransform;
        namespace Expression = rocRoller::Expression;
        using namespace Expression;

        /*
         * Code generation
         */
        struct CFCodeGeneratorVisitor
        {
            CFCodeGeneratorVisitor(KernelGraph                     graph,
                                   std::shared_ptr<Command>        command,
                                   std::shared_ptr<AssemblyKernel> kernel)
                : m_graph(graph)
                , m_command(command)
                , m_kernel(kernel)
                , m_context(kernel->context())
                , m_fastArith{kernel->context()}
            {
            }

            Generator<Instruction> generate()
            {
                auto coords = Transformer(
                    std::make_shared<HyperGraph>(m_graph.coordinates), m_context, m_fastArith);

                co_yield Instruction::Comment("CFCodeGeneratorVisitor::generate() begin");
                co_yield setup();
                std::set<TagType> candidates = {getTag(m_graph.control.getRootOperation())};

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

            /**
             *
             */
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

            bool hasGeneratedInputs(TagType const& tag)
            {
                auto inputTags = m_graph.control.getInputTags<ControlGraph::Sequence>(tag);
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
            Generator<Instruction> generate(std::set<TagType> candidates, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    concatenate("KernelGraph::CFCodeGeneratorVisitor::generate: ", candidates));

                co_yield Instruction::Comment(concatenate("generate(", candidates, ")"));

                while(!candidates.empty())
                {
                    std::set<TagType> nodes;

                    // Find all candidate nodes whose inputs have been satisfied
                    for(auto const& tag : candidates)
                        if(hasGeneratedInputs(tag))
                            nodes.insert(tag);

                    if(nodes.empty())
                    {
                        for(auto const& tag : candidates)
                            if(hasGeneratedInputs(tag))
                                nodes.insert(tag);
                    }

                    // If there are none, we have a problem.
                    AssertFatal(!nodes.empty(),
                                "Invalid control graph!",
                                ShowValue(m_graph.control),
                                ShowValue(candidates));

                    // Generate code for all the nodes we found.

                    std::vector<Generator<Instruction>> generators;
                    generators.reserve(nodes.size());

                    for(auto const& tag : nodes)
                    {
                        auto op = m_graph.control.getOperation(tag);
                        generators.push_back(call(op, coords));
                    }

                    auto proc      = Settings::getInstance()->get(Settings::Scheduler);
                    auto scheduler = Component::GetNew<Scheduling::Scheduler>(proc, m_context);
                    co_yield (*scheduler)(generators);

                    // Add output nodes to candidates.

                    for(auto const& tag : nodes)
                    {
                        auto outTags = m_graph.control.getOutputTags<ControlGraph::Sequence>(tag);
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

            Generator<Instruction> generate(std::vector<TagType> const& candidates,
                                            Transformer const&          coords)
            {
                std::set<TagType> cset(candidates.begin(), candidates.end());
                co_yield generate(cset, coords);
            }

            Generator<Instruction> call(ControlGraph::Operation operation,
                                        Transformer const&      coords)
            {
                auto opName = toString(operation);
                co_yield Instruction::Comment(opName + " BEGIN");

                auto opTag = getTag(operation);
                AssertFatal(m_completedControlNodes.find(opTag) == m_completedControlNodes.end(),
                            ShowValue(operation));

                co_yield std::visit(*this, operation, std::variant<Transformer>(coords));

                co_yield Instruction::Comment(opName + " END");

                m_completedControlNodes.insert(opTag);
            }

            Generator<Instruction> operator()(Operations::E_Neg const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Neg"));

                auto src = m_context->registerTagManager()->getRegister(xop.a);
                auto dst
                    = m_context->registerTagManager()->getRegister(xop.dest, src->placeholder());

                co_yield generateOp<Expression::Negate>(dst, src);
            }

            Generator<Instruction> operator()(Operations::E_Abs const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Abs"));
                // TODO: Finish codegen for E_Abs
                throw std::runtime_error("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Not const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Not"));
                // TODO: Finish codegen for E_Not
                throw std::runtime_error("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Add const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Add"));

                co_yield generateCommutativeBinaryOp<Expression::Add>(xop.dest, xop.a, xop.b);
            }

            Generator<Instruction> operator()(Operations::E_Sub const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Sub"));

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);
                auto dst
                    = m_context->registerTagManager()->getRegister(xop.dest, lhs->placeholder());

                co_yield generateOp<Expression::Subtract>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_Mul const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Mul"));

                co_yield generateCommutativeBinaryOp<Expression::Multiply>(xop.dest, xop.a, xop.b);
            }

            Generator<Instruction> operator()(Operations::E_Div const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Div"));

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);
                auto dst
                    = m_context->registerTagManager()->getRegister(xop.dest, lhs->placeholder());

                co_yield generateOp<Expression::Divide>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_And const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_And"));
                // TODO: Finish codegen for E_And
                throw std::runtime_error("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Or const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Or"));
                // TODO: Finish codegen for E_Or
                throw std::runtime_error("Not implemented yet.");
            }

            Generator<Instruction> operator()(ControlGraph::ElementOp const& edge,
                                              Transformer                    coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::ElementOp()");
                co_yield_(Instruction::Comment("GEN: ElementOp"));
                co_yield std::visit(
                    [&](auto&& arg) -> Generator<Instruction> { co_yield (*this)(arg); },
                    *edge.xop);
            }

            Generator<Instruction> operator()(ControlGraph::Kernel const& edge, Transformer coords)
            {
                co_yield Instruction::Comment("Begin Kernel");

                auto body = m_graph.control.getOutputTags<ControlGraph::Body>(getTag(edge));
                std::set<TagType> bodyNodes(body.begin(), body.end());
                co_yield generate(bodyNodes, coords);

                co_yield Instruction::Comment("End Kernel");

                co_return;
            }

            Generator<Instruction> operator()(ControlGraph::ForLoopOp const& edge,
                                              Transformer                    coords)
            {
                auto loopTag = getTag(edge);

                // TODO: Logging level for these comments.
                co_yield Instruction::Comment("For Loop Begin");

                auto topLabel = m_context->labelAllocator()->label("ForLoopTop");
                auto botLabel = m_context->labelAllocator()->label("ForLoopBottom");

                co_yield Instruction::Comment("Initialize For Loop");
                co_yield generate(m_graph.control.getOutputTags<ControlGraph::Initialize>(loopTag),
                                  coords);

                auto iterReg = m_context->registerTagManager()->getRegister(edge.counterTag.ctag);

                {
                    auto loopDims
                        = m_graph.coordinates.getOutputs(edge.counterTag, EdgeType::DataFlow);
                    for(auto const& dim : loopDims)
                    {
                        coords.setCoordinate(dim, iterReg->expression());
                    }
                }

                co_yield_(Instruction::Lock(Scheduling::Dependency::Branch, "Lock For Loop"));
                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(edge.condition);
                auto conditionResult = conditionRegisterType == Register::Type::Special
                                               && conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();
                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(edge.condition), m_context);
                co_yield m_context->brancher()->branchIfZero(
                    botLabel,
                    conditionResult,
                    concatenate("Condition: Top (jump to " + botLabel->toString() + " if false)"));

                co_yield Instruction::Label(topLabel);

                co_yield generate(m_graph.control.getOutputTags<ControlGraph::Body>(getTag(edge)),
                                  coords);

                co_yield Instruction::Comment("For Loop Increment");
                co_yield generate(
                    m_graph.control.getOutputTags<ControlGraph::ForLoopIncrement>(loopTag), coords);
                co_yield Instruction::Comment("Condition: Bottom (jump to " + topLabel->toString()
                                              + " if true)");

                co_yield Expression::generate(
                    conditionResult, coords.getTransducer()(edge.condition), m_context);
                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    conditionResult,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                co_yield Instruction::Label(botLabel);

                co_yield Instruction::Comment("For Loop End");
                co_yield_(Instruction::Unlock("Unlock For Loop"));
            }

            Generator<Instruction> operator()(ControlGraph::UnrollOp const& unroll,
                                              Transformer                   coords)
            {
                co_yield Instruction::Comment("Unroll Begin");
                int  unrollSize          = std::get<int>(evaluate(unroll.size));
                auto savedCompletedNodes = m_completedControlNodes;
                for(int i = 0; i < unrollSize; i++)
                {
                    m_completedControlNodes = savedCompletedNodes;
                    co_yield Instruction::Comment(concatenate("Unroll ", i, " of ", unrollSize));
                    auto unrollDims = m_graph.coordinates.getOutputs(getTag(Linear(unroll.tag)),
                                                                     EdgeType::DataFlow);
                    for(auto const& dim : unrollDims)
                    {
                        coords.setCoordinate(dim, literal(i));
                    }
                    co_yield generate(
                        m_graph.control.getOutputTags<ControlGraph::Body>(getTag(unroll)), coords);
                }
                co_yield Instruction::Comment("Unroll End");
            }

            Generator<Instruction> operator()(ControlGraph::Assign const& edge, Transformer coords)
            {
                auto varType = resultVariableType(edge.expression);

                auto dest = m_context->registerTagManager()->getRegister(
                    edge.destTag, edge.regType, varType, edge.valueCount);

                co_yield Expression::generate(
                    dest, coords.getTransducer()(edge.expression), m_context);
            }

            Generator<Instruction> operator()(ControlGraph::Barrier const& edge, Transformer coords)
            {
                co_yield m_context->mem()->barrier();
            }

            Generator<Instruction> operator()(ControlGraph::LoadLinear const& edge,
                                              Transformer                     coords)
            {
                throw std::runtime_error("LoadLinear present in kernel graph.");
            }

            Generator<Instruction> loadMacroTileVGPR(ControlGraph::LoadTiled const& load,
                                                     User const&                    user,
                                                     MacroTile const&               tiled,
                                                     Transformer                    coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadTiledWorkgroupVGPR()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileVGPR"));

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tiled.tag));
                auto vgpr  = MkVGPR(vtype, product(tiled.subTileSizes));
                vgpr       = m_context->registerTagManager()->getRegister(getTag(tiled).ctag, vgpr);
                co_yield vgpr->allocate();

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto bufDesc = BufferDescriptor(m_context);
                auto bufOpt  = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                auto const m = tiled.subTileSizes[0];
                auto const n = tiled.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                auto rowIndex = ThreadTileIndex(tiled.tag, 0);
                auto colIndex = ThreadTileIndex(tiled.tag, 1);

                coords.setCoordinate(rowIndex, L(0));
                coords.setCoordinate(colIndex, L(0));

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto rowOffset     = MkVGPR(DataType::Int64);
                auto rowOffsetExpr = rowOffset->expression();

                auto baseIndexExpr = coords.reverse({user})[0];
                co_yield generate(rowOffset, baseIndexExpr * L(numBytes));

                // row stride
                auto rowStride     = MkSGPR(DataType::Int64);
                auto rowStrideExpr = rowStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.reverseStride(rowIndex, L(1), {user})[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(rowStride, sgpr);
                }

                // col stride
                auto colStride     = MkSGPR(DataType::Int64);
                auto colStrideExpr = colStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.reverseStride(colIndex, L(1), {user})[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(colStride, sgpr);
                }

                auto increment     = MkSGPR(DataType::Int64);
                auto incrementExpr = increment->expression();
                co_yield copy(increment, s_ptr);

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            rowOffset->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                            co_yield bufDesc.incrementBasePointer(colStride);
                    }

                    if(i < m - 1)
                        co_yield generate(increment, incrementExpr + rowStrideExpr);
                    co_yield bufDesc.setBasePointer(increment);
                }
            }

            Generator<Instruction> loadMacroTileWAVE(ControlGraph::LoadTiled const& load,
                                                     User const&                    user,
                                                     WaveTile const&                tiled,
                                                     Transformer                    coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVE()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileWAVE"));

                // Move the argument pointer into v_ptr
                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto bufDesc = BufferDescriptor(m_context);
                auto bufOpt  = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                // Register Value used to contain the offset
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                uint num_elements = tiled.sizes[0] * tiled.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tiled.tag));

                if(vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr
                        = m_context->registerTagManager()->getRegister(getTag(tiled).ctag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    for(uint a = 0; a < num_vgpr; a += 2)
                    {
                        coords.setCoordinate(VGPR(tiled.tag), literal(a));

                        auto user_index = coords.reverse({user})[0];
                        setComment(user_index, "User Index Expression");
                        co_yield generateOffset(offset, user_index, vtype.dataType);

                        coords.setCoordinate(VGPR(tiled.tag), literal(a + 1));

                        auto user_index2 = coords.reverse({user})[0];
                        setComment(user_index2, "User Index Expression");
                        co_yield generateOffset(offset2, user_index2, vtype.dataType);

                        co_yield m_context->mem()->loadAndPackBuffer(
                            vgpr->element({static_cast<int>(a / 2)}),
                            offset,
                            offset2,
                            bufDesc,
                            bufOpt);
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, vtype, num_vgpr);

                    auto vgpr
                        = m_context->registerTagManager()->getRegister(getTag(tiled).ctag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    for(uint a = 0; a < num_vgpr; ++a)
                    {
                        coords.setCoordinate(VGPR(tiled.tag), literal(a));

                        auto user_index = coords.reverse({user})[0];
                        setComment(user_index, "User Index Expression");
                        co_yield generateOffset(offset, user_index, vgpr->variableType().dataType);

                        co_yield m_context->mem()->loadBuffer(vgpr->element({static_cast<int>(a)}),
                                                              offset->subset({0}),
                                                              0,
                                                              bufDesc,
                                                              bufOpt,
                                                              numBytes);
                    }
                }
            }

            Generator<Instruction> operator()(ControlGraph::LoadTiled const& load,
                                              Transformer                    coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadTiled()");
                co_yield_(Instruction::Comment("GEN: LoadTiled"));

                auto user = m_graph.coordinates.getDimension(User(load.tag));
                auto tile = m_graph.coordinates.getDimension(MacroTile(load.tag));

                switch(tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield loadMacroTileVGPR(load, user, tile, coords);
                    break;
                case MemoryType::WAVE:
                {
                    auto waveTile = m_graph.coordinates.getDimension(WaveTile(load.tag));
                    co_yield loadMacroTileWAVE(load, user, waveTile, coords);
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            /* LoadLDSTile edge looks like:
             *
             *             LoadLDSTile
             *   { LDS } ---------> { vgpr }
             */
            Generator<Instruction> operator()(ControlGraph::LoadLDSTile const& load,
                                              Transformer                      coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadLDSTile()");
                co_yield_(Instruction::Comment("GEN: LoadLDSTile"));

                auto user    = m_graph.coordinates.getDimension(User(load.tag));
                auto tile    = m_graph.coordinates.getDimension(MacroTile(load.tag));
                auto outputs = m_graph.coordinates.getOutputs(getTag(tile), EdgeType::DataFlow);
                AssertFatal(outputs.size() == 1, "Invalid LoadLDSTile MacroTile to LDS edge.");

                auto lds = std::get<LDS>(outputs[0]);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(getTag(lds).ctag);

                auto vtype = ldsAllocation->variableType();

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                coords.setCoordinate(Workitem(lds.tag, 0, nullptr, true), m_workitem[0]);
                coords.setCoordinate(Workitem(lds.tag, 1, nullptr, true), m_workitem[1]);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({lds})[0])
                      * Expression::literal(product(tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                co_yield generate(lds_offset,
                                  coords.getTransducer()(lds_offset_expr * L(numBytes)));

                auto vgpr = m_context->registerTagManager()->getRegister(getTag(tile).ctag);

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

            /* LoadVGPR edge looks like:
             *
             *             LoadVGPR
             *   { user } ---------> { vgpr }
             */
            Generator<Instruction> operator()(ControlGraph::LoadVGPR const& load,
                                              Transformer                   coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadVGPR()");
                co_yield_(Instruction::Comment("GEN: LoadVGPR"));
                auto user = m_graph.coordinates.getDimension(User(load.tag));
                auto vgpr = m_graph.coordinates.getDimension(VGPR(load.tag));
                auto dst  = m_context->registerTagManager()->getRegister(
                    user.tag,
                    Register::Type::Vector,
                    Operations::VariableTypeVisitor()(*m_command->findTag(user.tag)));
                co_yield Register::AllocateIfNeeded(dst);

                auto cmd = *m_command->findTag(load.tag);
                if(std::holds_alternative<Operations::T_Load_Scalar>(cmd))
                {
                    auto vtype = std::get<Operations::T_Load_Scalar>(cmd).variableType();
                    if(vtype.isPointer())
                        co_yield LoadVGPR_FromGlobalScalarPointer(user, dst, coords);
                    else
                        co_yield LoadVGPR_FromScalarValue(user, dst, load, coords);
                }
                else
                {
                    co_yield LoadVGPR_FromGlobalArray(user, vgpr, dst, coords);
                }
            }

            Generator<Instruction> LoadVGPR_FromScalarValue(User                             user,
                                                            std::shared_ptr<Register::Value> vgpr,
                                                            ControlGraph::LoadVGPR const&    op,
                                                            Transformer                      coords)
            {
                co_yield_(Instruction::Comment("GEN: LoadVGPR; scalar"));

                Register::ValuePtr s_value;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_value);
                co_yield m_context->copier()->copy(vgpr, s_value, "Move value");
            }

            Generator<Instruction> LoadVGPR_FromGlobalScalarPointer(
                User user, std::shared_ptr<Register::Value> vgpr, Transformer coords)
            {
                co_yield_(Instruction::Comment("GEN: LoadVGPR; scalar"));

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, v_ptr, nullptr, numBytes);
            }

            Generator<Instruction> LoadVGPR_FromGlobalArray(User                             user,
                                                            VGPR                             vgpr,
                                                            std::shared_ptr<Register::Value> dst,
                                                            Transformer                      coords)
            {
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                co_yield_(Instruction::Comment("GEN: LoadVGPR; user index"));

                auto indexes = coords.reverse({user});
                co_yield generateOffset(offset, indexes[0], dst->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                auto numBytes = DataTypeInfo::Get(dst->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, dst, v_ptr, offset, numBytes);
            }

            Generator<Instruction> operator()(ControlGraph::Multiply const& mult,
                                              Transformer                   coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Multiply()");
                co_yield_(Instruction::Comment("GEN: Multiply"));

                auto waveA  = m_graph.coordinates.getDimension(WaveTile(mult.a));
                auto waveB  = m_graph.coordinates.getDimension(WaveTile(mult.b));
                auto macA   = m_graph.coordinates.getDimension(MacroTile(mult.a));
                auto macB   = m_graph.coordinates.getDimension(MacroTile(mult.b));
                auto loadAB = m_graph.control.getOutputTags<ControlGraph::Body>(getTag(mult));

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto D = m_context->registerTagManager()->getRegister(
                    mult.tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto completed = m_completedControlNodes;

                // D is not initialized here

                uint const num_wave_tiles = macA.sizes[1] / waveA.sizes[1];
                for(uint k = 0; k < num_wave_tiles; k++)
                {
                    m_completedControlNodes = completed; // TODO: remove this?

                    // A WaveTile number; tall-skinny column block
                    coords.setCoordinate(waveA.tileNumber(1), literal(k));
                    // B WaveTile number; short-fat row block
                    coords.setCoordinate(waveB.tileNumber(0), literal(k));

                    co_yield generate(loadAB, coords);

                    waveA.vgpr = m_context->registerTagManager()->getRegister(waveA.tag);
                    waveB.vgpr = m_context->registerTagManager()->getRegister(waveB.tag);

                    Expression::ExpressionPtr A = std::make_shared<Expression::Expression>(
                        std::make_shared<WaveTile>(waveA));
                    Expression::ExpressionPtr B = std::make_shared<Expression::Expression>(
                        std::make_shared<WaveTile>(waveB));

                    co_yield generate(D,
                                      std::make_shared<Expression::Expression>(
                                          Expression::MatrixMultiply(A, B, D->expression())));

                    // TODO: Extend MatrixMultiply arithmetic to recognize this
                    // co_yield generate(D, D + A * B);
                }
            }

            Generator<Instruction> operator()(ControlGraph::TensorContraction const& mul,
                                              Transformer                            coords)
            {
                Throw<FatalError>("TensorContraction present in kernel graph.");
            }

            Generator<Instruction> operator()(ControlGraph::StoreLinear const& edge,
                                              Transformer                      coords)
            {
                throw std::runtime_error("StoreLinear present in kernel graph.");
            }

            Generator<Instruction> storeMacroTileVGPR(ControlGraph::StoreTiled const& store,
                                                      User const&                     user,
                                                      MacroTile const&                tiled,
                                                      Transformer                     coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileVGPR()");
                co_yield_(Instruction::Comment("GEN: storeMacroTileVGPR"));

                auto vgpr = m_context->registerTagManager()->getRegister(getTag(tiled).ctag);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                auto bufDesc  = BufferDescriptor(m_context);
                auto bufOpt   = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                auto const m        = tiled.subTileSizes[0];
                auto const n        = tiled.subTileSizes[1];
                auto       rowIndex = ThreadTileIndex(tiled.tag, 0, true);
                auto       colIndex = ThreadTileIndex(tiled.tag, 1, true);

                coords.setCoordinate(rowIndex, L(0));
                coords.setCoordinate(colIndex, L(0));

                auto rowOffset     = MkVGPR(DataType::Int64);
                auto rowOffsetExpr = rowOffset->expression();

                auto baseIndexExpr = coords.forward({user})[0];
                co_yield generate(rowOffset, baseIndexExpr * L(numBytes));

                // row stride
                auto rowStride     = MkSGPR(DataType::Int64);
                auto rowStrideExpr = rowStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.forwardStride(rowIndex, L(1), {user})[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(rowStride, sgpr);
                }

                // col stride
                auto colStride     = MkSGPR(DataType::Int64);
                auto colStrideExpr = colStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.forwardStride(colIndex, L(1), {user})[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(colStride, sgpr);
                }

                auto increment     = MkSGPR(DataType::Int64);
                auto incrementExpr = increment->expression();
                co_yield copy(increment, s_ptr);

                // TODO multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->storeBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            rowOffset->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield bufDesc.incrementBasePointer(colStride);
                        }
                    }
                    if(i < m - 1)
                    {
                        co_yield generate(increment, incrementExpr + rowStrideExpr);
                    }
                    co_yield bufDesc.setBasePointer(increment);
                }
            }

            Generator<Instruction> storeMacroTileWAVE(ControlGraph::StoreTiled const& store,
                                                      User const&                     user,
                                                      WaveTile const&                 tile,
                                                      Transformer                     coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVE()");
                co_yield_(Instruction::Comment("GEN: storeMacroTileWAVE"));

                uint num_elements = tile.sizes[0] * tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto agpr = m_context->registerTagManager()->getRegister(getTag(tile).ctag);

                AssertFatal(agpr->registerCount() == num_vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto bufDesc = BufferDescriptor(m_context);
                auto bufOpt  = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                auto value = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, agpr->variableType(), 1);
                co_yield value->allocate();

                auto dataType = store.dataType;
                auto converted
                    = Register::Value::Placeholder(m_context, Register::Type::Vector, dataType, 1);
                co_yield converted->allocate();

                auto numBytes = DataTypeInfo::Get(dataType).elementSize;

                for(uint a = 0; a < num_vgpr; ++a)
                {
                    coords.setCoordinate(VGPR(tile.tag, true), literal(a));

                    auto userIndex = coords.forward({user})[0];
                    co_yield generateOffset(offset, userIndex, dataType);
                    if(value->variableType() != dataType)
                    {
                        co_yield m_context->copier()->copy(value,
                                                           agpr->element({static_cast<int>(a)}));
                        co_yield Expression::generate(
                            converted,
                            convert(dataType, std::make_shared<Expression::Expression>(value)),
                            m_context);
                    }
                    else
                    {
                        co_yield m_context->copier()->copy(converted,
                                                           agpr->element({static_cast<int>(a)}));
                    }
                    //v_accvgpr_read for V_MFMA with 2 pass instructions is 4, 10 for 8 pass instruction
                    // and 18 for 16 pass instructions like V_MFMA_F32_32x32x8F16
                    co_yield Instruction::Nop(18, "Nops required for v_accvgpr_read");
                    co_yield m_context->mem()->storeBuffer(
                        converted, offset->subset({0}), 0, bufDesc, bufOpt, numBytes);
                }
            }

            Generator<Instruction> operator()(ControlGraph::StoreTiled const& store,
                                              Transformer                     coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreTiled()");
                co_yield_(Instruction::Comment("GEN: StoreTiled"));

                auto user = m_graph.coordinates.getDimension(User(store.tag, true));
                auto tile = m_graph.coordinates.getDimension(MacroTile(store.tag, 0, true));

                switch(tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(store, user, tile, coords);
                    break;
                case MemoryType::WAVE:
                {
                    auto waveTile = m_graph.coordinates.getDimension(WaveTile(store.tag, 2, true));
                    co_yield storeMacroTileWAVE(store, user, waveTile, coords);
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet.");
                }
            }

            Generator<Instruction> operator()(ControlGraph::StoreLDSTile const& store,
                                              Transformer                       coords)
            {
                co_yield_(Instruction::Comment("GEN: StoreLDSTile"));

                auto lds    = m_graph.coordinates.getDimension(LDS(store.tag));
                auto inputs = m_graph.coordinates.getInputs(getTag(lds), EdgeType::DataFlow);
                AssertFatal(inputs.size() == 1, "Invalid StoreLDSTile MacroTile to LDS edge.");

                auto tile = std::get<MacroTile>(inputs[0]);

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tile.tag));

                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                auto ldsAllocation = Register::Value::AllocateLDS(
                    m_context, vtype, product(tile.subTileSizes) * product(m_workgroupSize));

                m_context->registerTagManager()->addRegister(getTag(lds).ctag, ldsAllocation);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                coords.setCoordinate(Workitem(lds.tag, 0, nullptr, true), m_workitem[0]);
                coords.setCoordinate(Workitem(lds.tag, 1, nullptr, true), m_workitem[1]);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({lds})[0])
                      * Expression::literal(product(tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;
                co_yield generate(lds_offset,
                                  coords.getTransducer()(lds_offset_expr * L(numBytes)));

                // Temporary register that is used to copy the data from global memory to
                // local memory.
                auto vgpr = m_context->registerTagManager()->getRegister(getTag(tile).ctag);

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

            Generator<Instruction> operator()(ControlGraph::StoreVGPR const& store,
                                              Transformer                    coords)
            {
                co_yield_(Instruction::Comment("GEN: StoreVGPR"));
                co_yield_(Instruction::Comment("GEN: StoreVGPR; user index"));

                auto vgpr = VGPR(store.tag);
                auto src  = m_context->registerTagManager()->getRegister(store.tag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                coords.setCoordinate(Workitem(store.tag, 0, nullptr, true), m_workitem[0]);

                auto user    = m_graph.coordinates.getDimension(User(store.tag, true));
                auto indexes = coords.forward({user});

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
            KernelGraph                     m_graph;
            std::shared_ptr<Command>        m_command;
            std::shared_ptr<Context>        m_context;
            std::shared_ptr<AssemblyKernel> m_kernel;

            std::set<TagType> m_completedControlNodes;

            std::vector<ExpressionPtr> m_workgroup;
            std::vector<ExpressionPtr> m_workitem;
            std::vector<unsigned int>  m_workgroupSize;
            FastArithmetic             m_fastArith;
        };

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

                    if(nodes.empty())
                    {
                        for(auto const& tag : candidates)
                            if(hasGeneratedInputs(tag))
                                nodes.insert(tag);
                    }

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
                co_yield_(Instruction::Comment("GEN: E_Neg"));

                auto src = m_context->registerTagManager()->getRegister(a);
                auto dst = m_context->registerTagManager()->getRegister(tag, src->placeholder());

                co_yield generateOp<Expression::Negate>(dst, src);
            }

            Generator<Instruction> operator()(Operations::E_Abs, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Abs"));
                // TODO: Finish codegen for E_Abs
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Not, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Not"));
                // TODO: Finish codegen for E_Not
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Add, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Add"));

                co_yield generateCommutativeBinaryOp<Expression::Add>(tag, a, b);
            }

            Generator<Instruction> operator()(Operations::E_Sub, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Sub"));

                auto lhs = m_context->registerTagManager()->getRegister(a);
                auto rhs = m_context->registerTagManager()->getRegister(b);
                auto dst = m_context->registerTagManager()->getRegister(tag, lhs->placeholder());

                co_yield generateOp<Expression::Subtract>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_Mul, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Mul"));

                co_yield generateCommutativeBinaryOp<Expression::Multiply>(tag, a, b);
            }

            Generator<Instruction> operator()(Operations::E_Div, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Div"));

                auto lhs = m_context->registerTagManager()->getRegister(a);
                auto rhs = m_context->registerTagManager()->getRegister(b);
                auto dst = m_context->registerTagManager()->getRegister(tag, lhs->placeholder());

                co_yield generateOp<Expression::Divide>(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_And, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_And"));
                // TODO: Finish codegen for E_And
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(Operations::E_Or, int tag, int a, int b)
            {
                co_yield_(Instruction::Comment("GEN: E_Or"));
                // TODO: Finish codegen for E_Or
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(int                          tag,
                                              ControlHypergraph::ElementOp eop,
                                              CoordGraph::Transformer      coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::ElementOp()");
                co_yield_(Instruction::Comment("GEN: ElementOp"));

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
                co_yield Instruction::Comment("BEGIN SCOPE");

                auto scope = std::make_shared<ScopeManager>(m_context);
                coords.setScope(scope);

                auto body = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                .to<std::set>();
                co_yield generate(body, coords);

                scope->release();
                co_yield Instruction::Comment("END SCOPE");
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::ForLoopOp const& edge,
                                              CoordGraph::Transformer             coords)
            {
                Throw<FatalError>("Not implemented yet.");
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
                auto varType = resultVariableType(assign.expression);

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid Assign operation; coordinate missing.");
                auto dim_tag = connections[0].coordinate;

                auto dest = m_context->registerTagManager()->getRegister(
                    dim_tag, assign.regType, varType, assign.valueCount);

                auto scope = coords.getScope();
                if(scope)
                {
                    scope->add(dim_tag);
                }

                co_yield Expression::generate(dest, assign.expression, m_context);
            }

            Generator<Instruction>
                operator()(int, ControlHypergraph::Barrier const&, CoordGraph::Transformer)
            {
                co_yield m_context->mem()->barrier();
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
                co_yield_(Instruction::Comment("GEN: loadMacroTileVGPR"));

                auto [user_tag, user]         = m_graph.getDimension<CoordGraph::User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                auto vgpr = MkVGPR(load.vtype, product(mac_tile.subTileSizes));
                vgpr      = m_context->registerTagManager()->getRegister(mac_tile_tag, vgpr);
                co_yield vgpr->allocate();

                auto rowOffset = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), rowOffset);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                auto bufDesc  = BufferDescriptor(m_context);
                auto bufOpt   = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(rowOffset);

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                auto [row_index_tag, row_index]
                    = m_graph.getDimension<CoordGraph::ThreadTileIndex>(tag, 0);
                auto [col_index_tag, col_index]
                    = m_graph.getDimension<CoordGraph::ThreadTileIndex>(tag, 1);

                coords.setCoordinate(row_index_tag, L(0));
                coords.setCoordinate(col_index_tag, L(0));

                auto threadOffset = MkVGPR(DataType::Int64);
                {
                    auto expr = coords.reverse({user_tag})[0];
                    co_yield generate(threadOffset, expr * L(numBytes));
                }

                auto rowStride = MkSGPR(DataType::Int64);
                {
                    auto expr = coords.reverseStride(row_index_tag, L(1), {user_tag})[0];
                    co_yield generate(rowStride, expr * L(numBytes));
                }

                auto colStride = MkSGPR(DataType::Int64);
                {
                    auto expr = coords.reverseStride(col_index_tag, L(1), {user_tag})[0];
                    co_yield generate(colStride, expr * L(numBytes));
                }

                auto rowOffsetExpr = rowOffset->expression();
                auto rowStrideExpr = rowStride->expression();

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            threadOffset->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                            co_yield bufDesc.incrementBasePointer(colStride);
                    }

                    if(i < m - 1)
                        co_yield generate(rowOffset, rowOffsetExpr + rowStrideExpr);
                    co_yield bufDesc.setBasePointer(rowOffset);
                }
            }

            Generator<Instruction> loadMacroTileWAVE(int                                 tag,
                                                     ControlHypergraph::LoadTiled const& load,
                                                     CoordGraph::Transformer             coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVE()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileWAVE"));

                auto [user_tag, user]           = m_graph.getDimension<CoordGraph::User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<CoordGraph::WaveTile>(tag);
                auto [vgpr_tag, vgpr]           = m_graph.getDimension<CoordGraph::VGPR>(tag);

                // Move the argument pointer into v_ptr
                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto bufDesc = BufferDescriptor(m_context);
                auto bufOpt  = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                // Register Value used to contain the offset
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                if(load.vtype == DataType::Half)
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);

                    auto vgpr = m_context->registerTagManager()->getRegister(wave_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Int64, 1);

                    for(uint a = 0; a < num_vgpr; a += 2)
                    {
                        coords.setCoordinate(vgpr_tag, literal(a));

                        auto user_index = coords.reverse({user_tag})[0];
                        setComment(user_index, "User Index Expression");
                        co_yield generateOffset(offset, user_index, load.vtype.dataType);

                        coords.setCoordinate(vgpr_tag, literal(a + 1));

                        auto user_index2 = coords.reverse({user_tag})[0];
                        setComment(user_index2, "User Index Expression");
                        co_yield generateOffset(offset2, user_index2, load.vtype.dataType);

                        co_yield m_context->mem()->loadAndPackBuffer(
                            vgpr->element({static_cast<int>(a / 2)}),
                            offset,
                            offset2,
                            bufDesc,
                            bufOpt);
                    }
                }
                else
                {
                    auto tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);

                    auto vgpr = m_context->registerTagManager()->getRegister(wave_tile_tag, tmpl);
                    co_yield Register::AllocateIfNeeded(vgpr);

                    auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                    for(uint a = 0; a < num_vgpr; ++a)
                    {
                        coords.setCoordinate(vgpr_tag, literal(a));

                        auto user_index = coords.reverse({user_tag})[0];
                        setComment(user_index, "User Index Expression");
                        co_yield generateOffset(offset, user_index, vgpr->variableType().dataType);

                        co_yield m_context->mem()->loadBuffer(vgpr->element({static_cast<int>(a)}),
                                                              offset->subset({0}),
                                                              0,
                                                              bufDesc,
                                                              bufOpt,
                                                              numBytes);
                    }
                }
            }

            Generator<Instruction> operator()(int                                 tag,
                                              ControlHypergraph::LoadTiled const& load,
                                              CoordGraph::Transformer             coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadTiled()");
                co_yield_(Instruction::Comment("GEN: LoadTiled"));

                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield loadMacroTileVGPR(tag, load, coords);
                    break;
                case MemoryType::WAVE:
                    co_yield loadMacroTileWAVE(tag, load, coords);
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
                co_yield_(Instruction::Comment("GEN: LoadLDSTile"));

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
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadVGPR()");
                co_yield_(Instruction::Comment("GEN: LoadVGPR"));

                auto [userTag, user] = m_graph.getDimension<CoordGraph::User>(tag);
                auto [vgprTag, vgpr] = m_graph.getDimension<CoordGraph::VGPR>(tag);

                auto dst = m_context->registerTagManager()->getRegister(
                    vgprTag, Register::Type::Vector, load.vtype.dataType);
                co_yield Register::AllocateIfNeeded(dst);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): dst tag {}", vgprTag);

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
                co_yield_(Instruction::Comment("GEN: LoadVGPR; scalar value"));

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
                co_yield_(Instruction::Comment("GEN: LoadVGPR; scalar pointer"));

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

                co_yield_(Instruction::Comment("GEN: LoadVGPR; user index"));

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
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::Multiply()");
                co_yield_(Instruction::Comment("GEN: Multiply"));

                auto [waveA_tag, waveA] = m_graph.getDimension<CoordGraph::WaveTile>(mult.a);
                auto [waveB_tag, waveB] = m_graph.getDimension<CoordGraph::WaveTile>(mult.b);
                auto [macA_tag, macA]   = m_graph.getDimension<CoordGraph::MacroTile>(mult.a);
                auto [macB_tag, macB]   = m_graph.getDimension<CoordGraph::MacroTile>(mult.b);
                auto [n_waveA_y_tag, n_waveA_y]
                    = m_graph.getDimension<CoordGraph::WaveTileNumber>(mult.a, 1);
                auto [n_waveB_x_tag, n_waveB_x]
                    = m_graph.getDimension<CoordGraph::WaveTileNumber>(mult.b, 0);

                auto loadAB = m_graph.control.getOutputNodeIndices<ControlHypergraph::Body>(tag)
                                  .to<std::set>();

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto D = m_context->registerTagManager()->getRegister(
                    tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto completed = m_completedControlNodes;

                // D is not initialized here

                uint const num_wave_tiles = macA.sizes[1] / waveA.sizes[1];
                for(uint k = 0; k < num_wave_tiles; k++)
                {
                    m_completedControlNodes = completed; // TODO: remove this?

                    // A WaveTile number; tall-skinny column block
                    coords.setCoordinate(n_waveA_y_tag, literal(k));
                    // B WaveTile number; short-fat row block
                    coords.setCoordinate(n_waveB_x_tag, literal(k));

                    co_yield generate(loadAB, coords);

                    waveA.vgpr = m_context->registerTagManager()->getRegister(waveA_tag);
                    waveB.vgpr = m_context->registerTagManager()->getRegister(waveB_tag);

                    Expression::ExpressionPtr A = std::make_shared<Expression::Expression>(
                        std::make_shared<CoordGraph::WaveTile>(waveA));
                    Expression::ExpressionPtr B = std::make_shared<Expression::Expression>(
                        std::make_shared<CoordGraph::WaveTile>(waveB));

                    co_yield generate(D,
                                      std::make_shared<Expression::Expression>(
                                          Expression::MatrixMultiply(A, B, D->expression())));
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
                co_yield_(Instruction::Comment("GEN: storeMacroTileVGPR"));

                auto [user_tag, user]         = m_graph.getDimension<CoordGraph::User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                auto rowOffset = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), rowOffset);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                auto bufDesc  = BufferDescriptor(m_context);
                auto bufOpt   = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(rowOffset);

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                auto [row_index_tag, row_index]
                    = m_graph.getDimension<CoordGraph::ThreadTileIndex>(tag, 0);
                auto [col_index_tag, col_index]
                    = m_graph.getDimension<CoordGraph::ThreadTileIndex>(tag, 1);

                coords.setCoordinate(row_index_tag, L(0));
                coords.setCoordinate(col_index_tag, L(0));

                auto threadOffset = MkVGPR(DataType::Int64);
                {
                    auto expr = coords.forward({user_tag})[0];
                    co_yield generate(threadOffset, expr * L(numBytes));
                }

                auto rowStride = MkSGPR(DataType::Int64);
                {
                    auto expr = coords.forwardStride(row_index_tag, L(1), {user_tag})[0];
                    co_yield generate(rowStride, expr * L(numBytes));
                }

                auto colStride = MkSGPR(DataType::Int64);
                {
                    auto expr = coords.forwardStride(col_index_tag, L(1), {user_tag})[0];
                    co_yield generate(colStride, expr * L(numBytes));
                }

                auto rowOffsetExpr = rowOffset->expression();
                auto rowStrideExpr = rowStride->expression();

                // TODO multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->storeBuffer(
                            vgpr->element({static_cast<int>(i * n + j)}),
                            threadOffset->subset({0}),
                            0,
                            bufDesc,
                            bufOpt,
                            numBytes);
                        if(j < n - 1)
                        {
                            co_yield bufDesc.incrementBasePointer(colStride);
                        }
                    }
                    if(i < m - 1)
                    {
                        co_yield generate(rowOffset, rowOffsetExpr + rowStrideExpr);
                    }
                    co_yield bufDesc.setBasePointer(rowOffset);
                }
            }

            Generator<Instruction> storeMacroTileWAVE(int                                  tag,
                                                      ControlHypergraph::StoreTiled const& store,
                                                      CoordGraph::Transformer              coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVE()");
                co_yield_(Instruction::Comment("GEN: storeMacroTileWAVE"));

                auto [user_tag, user]           = m_graph.getDimension<CoordGraph::User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<CoordGraph::WaveTile>(tag);
                auto [vgpr_tag, vgpr]           = m_graph.getDimension<CoordGraph::VGPR>(tag);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto agpr = m_context->registerTagManager()->getRegister(vgpr_tag);

                AssertFatal(agpr->registerCount() == num_vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto bufDesc = BufferDescriptor(m_context);
                auto bufOpt  = BufferInstructionOptions();

                co_yield bufDesc.setup();
                co_yield bufDesc.setBasePointer(s_ptr);

                auto offset = MkVGPR(DataType::Int64);
                auto value  = MkVGPR(agpr->variableType());

                auto converted = MkVGPR(store.dataType);

                auto numBytes = DataTypeInfo::Get(store.dataType).elementSize;

                for(uint a = 0; a < num_vgpr; ++a)
                {
                    // TODO Use a stride here
                    coords.setCoordinate(vgpr_tag, literal(a));

                    auto user_index = coords.forward({user_tag})[0];
                    co_yield generateOffset(offset, user_index, store.dataType);
                    if(value->variableType() != store.dataType)
                    {
                        co_yield m_context->copier()->copy(value,
                                                           agpr->element({static_cast<int>(a)}));
                        co_yield Expression::generate(
                            converted,
                            convert(store.dataType,
                                    std::make_shared<Expression::Expression>(value)),
                            m_context);
                    }
                    else
                    {
                        co_yield m_context->copier()->copy(converted,
                                                           agpr->element({static_cast<int>(a)}));
                    }
                    //v_accvgpr_read for V_MFMA with 2 pass instructions is 4, 10 for 8 pass instruction
                    // and 18 for 16 pass instructions like V_MFMA_F32_32x32x8F16
                    co_yield Instruction::Nop(18, "Nops required for v_accvgpr_read");
                    co_yield m_context->mem()->storeBuffer(
                        converted, offset->subset({0}), 0, bufDesc, bufOpt, numBytes);
                }
            }

            Generator<Instruction> operator()(int                                  tag,
                                              ControlHypergraph::StoreTiled const& store,
                                              CoordGraph::Transformer              coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreTiled()");
                co_yield_(Instruction::Comment("GEN: StoreTiled"));

                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<CoordGraph::MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(tag, store, coords);
                    break;
                case MemoryType::WAVE:
                    co_yield storeMacroTileWAVE(tag, store, coords);
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
                co_yield_(Instruction::Comment("GEN: StoreLDSTile"));

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
                co_yield_(Instruction::Comment("GEN: StoreVGPR"));

                auto [vgprTag, vgpr] = m_graph.getDimension<CoordGraph::VGPR>(tag);
                auto [userTag, user] = m_graph.getDimension<CoordGraph::User>(tag);

                auto src = m_context->registerTagManager()->getRegister(vgprTag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto indexes = coords.forward({userTag});

                co_yield_(Instruction::Comment("GEN: StoreVGPR; user index"));
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

        Generator<Instruction> generate(KernelGraph                     graph,
                                        std::shared_ptr<Command>        command,
                                        std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::generate");
            rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}", graph.toDOT());

            auto visitor = CFCodeGeneratorVisitor(graph, command, kernel);

            co_yield visitor.generate();
        }

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
