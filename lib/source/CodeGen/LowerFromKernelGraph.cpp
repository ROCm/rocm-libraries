
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterUtils.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/Timer.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <iostream>
#include <memory>
#include <set>
#include <variant>

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

                    for(auto const& tag : nodes)
                    {
                        auto op = m_graph.control.getOperation(tag);
                        co_yield (*this)(op, coords);
                    }

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

            Generator<Instruction> generate(std::vector<TagType> const& candidates,
                                            Transformer const&          coords)
            {
                std::set<TagType> cset(candidates.begin(), candidates.end());
                co_yield generate(cset, coords);
            }

            Generator<Instruction> operator()(ControlGraph::Operation const& operation,
                                              Transformer const&             coords)
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

                auto arith = Arithmetic::Get(dst);

                co_yield arith->negate(dst, src);
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

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);

                AssertFatal(lhs->valueCount() == rhs->valueCount(), "E_Add size mismatch.");

                auto regType    = Register::Type::Vector;
                auto varType    = VariableType::Promote(lhs->variableType(), rhs->variableType());
                auto valueCount = lhs->valueCount();

                co_yield m_context->copier()->ensureType(lhs, lhs, regType);
                co_yield m_context->copier()->ensureType(rhs, rhs, regType);

                auto dst = m_context->registerTagManager()->getRegister(
                    xop.dest, regType, varType, valueCount);
                co_yield Register::AllocateIfNeeded(dst);

                auto arith = Arithmetic::Get(dst);

                for(int i = 0; i < dst->valueCount(); ++i)
                    co_yield arith->add(dst->element({i}), lhs->element({i}), rhs->element({i}));
            }

            Generator<Instruction> operator()(Operations::E_Sub const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Sub"));

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);
                auto dst
                    = m_context->registerTagManager()->getRegister(xop.dest, lhs->placeholder());

                auto arith = Arithmetic::Get(dst);

                co_yield arith->sub(dst, lhs, rhs);
            }

            Generator<Instruction> operator()(Operations::E_Mul const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Mul"));

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);

                auto regType    = Register::Type::Vector;
                auto varType    = VariableType::Promote(lhs->variableType(), rhs->variableType());
                auto valueCount = std::max(lhs->valueCount(), rhs->valueCount());

                co_yield m_context->copier()->ensureType(lhs, lhs, regType);
                co_yield m_context->copier()->ensureType(rhs, rhs, regType);

                auto dst = m_context->registerTagManager()->getRegister(
                    xop.dest, regType, varType, valueCount);
                co_yield Register::AllocateIfNeeded(dst);

                auto arith = Arithmetic::Get(dst);

                if(lhs->valueCount() == rhs->valueCount())
                {
                    for(int k = 0; k < lhs->valueCount(); ++k)
                    {
                        co_yield arith->mul(
                            dst->element({k}), lhs->element({k}), rhs->element({k}));
                    }
                }
                else if(lhs->valueCount() == 1)
                {
                    for(int k = 0; k < rhs->valueCount(); ++k)
                    {
                        co_yield arith->mul(dst->element({k}), lhs, rhs->element({k}));
                    }
                }
                else if(rhs->valueCount() == 1)
                {
                    for(int k = 0; k < lhs->valueCount(); ++k)
                    {
                        co_yield arith->mul(dst->element({k}), lhs->element({k}), rhs);
                    }
                }
                else
                {
                    Throw<FatalError>("Multiplication must be scalar or element-wise.");
                }
            }

            Generator<Instruction> operator()(Operations::E_Div const& xop)
            {
                co_yield_(Instruction::Comment("GEN: E_Div"));

                auto lhs = m_context->registerTagManager()->getRegister(xop.a);
                auto rhs = m_context->registerTagManager()->getRegister(xop.b);
                auto dst
                    = m_context->registerTagManager()->getRegister(xop.dest, lhs->placeholder());

                auto arith = Arithmetic::Get(dst);

                co_yield arith->div(dst, lhs, rhs);
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

                auto scc = m_context->getSCC();
                co_yield Expression::generate(scc, edge.condition, m_context);
                co_yield m_context->brancher()->branchIfZero(
                    botLabel,
                    scc,
                    concatenate("Condition: Top (jump to " + botLabel->toString() + " if false)"));

                co_yield Instruction::Label(topLabel);

                co_yield generate(m_graph.control.getOutputTags<ControlGraph::Body>(getTag(edge)),
                                  coords);

                co_yield Instruction::Comment("For Loop Increment");
                co_yield generate(
                    m_graph.control.getOutputTags<ControlGraph::ForLoopIncrement>(loopTag), coords);
                co_yield Instruction::Comment("Condition: Bottom (jump to " + topLabel->toString()
                                              + " if true)");

                co_yield Expression::generate(scc, edge.condition, m_context);
                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    scc,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                co_yield Instruction::Label(botLabel);

                co_yield Instruction::Comment("For Loop End");
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

                co_yield Expression::generate(dest, edge.expression, m_context);
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

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield copy(v_ptr, s_ptr);

                auto const m = tiled.subTileSizes[0];
                auto const n = tiled.subTileSizes[1];

                auto rowIndex = ThreadTileIndex(tiled.tag, 0);
                auto colIndex = ThreadTileIndex(tiled.tag, 1);

                coords.setCoordinate(rowIndex, L(0));
                coords.setCoordinate(colIndex, L(0));

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto rowOffset     = MkVGPR(DataType::Int64);
                auto rowOffsetExpr = rowOffset->expression();
                auto offset        = MkVGPR(DataType::Int64);
                auto offsetExpr    = offset->expression();

                auto baseIndexExpr = coords.reverse({user})[0];
                co_yield generate(rowOffset, baseIndexExpr * L(numBytes));

                // row stride
                auto rowStride     = MkVGPR(DataType::Int64);
                auto rowStrideExpr = rowStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.reverseStride(rowIndex, L(1), {user}, simplify)[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(rowStride, sgpr);
                }

                // col stride
                auto colStride     = MkVGPR(DataType::Int64);
                auto colStrideExpr = colStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.reverseStride(colIndex, L(1), {user}, simplify)[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(colStride, sgpr);
                }

                // TODO multi dimensional tiles
                for(uint i = 0; i < m; ++i)
                {
                    co_yield copy(offset, rowOffset);
                    for(uint j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->load(
                            MemoryInstructions::MemoryKind::Flat,
                            vgpr->element({static_cast<int>(i * n + j)}),
                            v_ptr,
                            offset,
                            numBytes);

                        if(j < n - 1)
                            co_yield generate(offset, offsetExpr + colStrideExpr);
                    }

                    if(i < m - 1)
                        co_yield generate(rowOffset, rowOffsetExpr + rowStrideExpr);
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

                uint num_elements = tiled.sizes[0] * tiled.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tiled.tag));
                auto tmpl  = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, vtype, num_vgpr);

                auto vgpr = m_context->registerTagManager()->getRegister(getTag(tiled).ctag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);

                co_yield m_context->copier()->copy(v_ptr, s_ptr);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                auto numBytes = (uint)DataTypeInfo::Get(vgpr->variableType()).elementSize;

                for(uint a = 0; a < num_vgpr; ++a)
                {
                    coords.setCoordinate(VGPR(tiled.tag), literal(a));

                    auto user_indexes = coords.reverse({user});
                    auto user_index
                        = fastDivision(fastMultiplication(simplify(user_indexes[0])), m_context);
                    co_yield generateOffset(offset, user_index, vgpr->variableType().dataType);

                    co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Flat,
                                                    vgpr->element({static_cast<int>(a)}),
                                                    v_ptr,
                                                    offset,
                                                    numBytes);
                }
            }

            Generator<Instruction> operator()(ControlGraph::LoadTiled const& load,
                                              Transformer                    coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::LoadTiled()");
                co_yield_(Instruction::Comment("GEN: LoadTiled"));

                switch(load.tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield loadMacroTileVGPR(load, load.user, load.tile, coords);
                    break;
                case MemoryType::WAVE:
                {
                    auto waveTile = m_graph.coordinates.getDimension(WaveTile(load.tag));
                    co_yield loadMacroTileWAVE(load, load.user, waveTile, coords);
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

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation
                    = m_context->registerTagManager()->getRegister(getTag(load.lds).ctag);

                auto vtype = ldsAllocation->variableType();

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                coords.setCoordinate(Workitem(load.lds.tag, 0, nullptr, true), m_workitem[0]);
                coords.setCoordinate(Workitem(load.lds.tag, 1, nullptr, true), m_workitem[1]);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({load.lds})[0])
                      * Expression::literal(product(load.tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;

                co_yield generate(lds_offset,
                                  Expression::simplify(Expression::fuse(
                                      m_fastArith(lds_offset_expr * L(numBytes)))));

                auto vgpr = m_context->registerTagManager()->getRegister(getTag(load.tile).ctag);

                auto const m = load.tile.subTileSizes[0];
                auto const n = load.tile.subTileSizes[1];

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->loadLocal(vgpr->element({i * n + j}),
                                                             lds_offset,
                                                             std::to_string((i * n + j) * numBytes),
                                                             numBytes);
                    }
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
                auto user = load.user;
                auto vgpr = VGPR(user.tag);
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

                waveA.vgpr = m_context->registerTagManager()->getRegister(
                    waveA.tag, Register::Type::Vector, DataType::Float, 1);
                waveB.vgpr = m_context->registerTagManager()->getRegister(
                    waveB.tag, Register::Type::Vector, DataType::Float, 1);

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto A
                    = std::make_shared<Expression::Expression>(std::make_shared<WaveTile>(waveA));
                auto B
                    = std::make_shared<Expression::Expression>(std::make_shared<WaveTile>(waveB));
                auto D = m_context->registerTagManager()->getRegister(
                    mult.tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto mfma = Component::Get<rocRoller::InstructionGenerators::MatrixMultiply>(
                    rocRoller::InstructionGenerators::MatrixMultiply::Argument{
                        m_context,
                        D->variableType().dataType,
                        waveA.vgpr->variableType().dataType});

                auto completed = m_completedControlNodes;

                // D is not initialised here

                uint const num_wave_tiles = macA.sizes[1] / waveA.sizes[1];
                for(uint k = 0; k < num_wave_tiles; k++)
                {
                    m_completedControlNodes = completed; // TODO: remove this?

                    // A WaveTile number; tall-skinny column block
                    coords.setCoordinate(waveA.tileNumber(1), literal(k));
                    // B WaveTile number; short-fat row block
                    coords.setCoordinate(waveB.tileNumber(0), literal(k));

                    co_yield generate(loadAB, coords);
                    co_yield mfma->mul(D,
                                       waveA.vgpr,
                                       waveB.vgpr,
                                       waveA.sizes[0],
                                       waveB.sizes[1],
                                       waveA.sizes[1],
                                       1);

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

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield copy(v_ptr, s_ptr);

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;

                auto const m = tiled.subTileSizes[0];
                auto const n = tiled.subTileSizes[1];

                auto rowIndex = ThreadTileIndex(tiled.tag, 0, true);
                auto colIndex = ThreadTileIndex(tiled.tag, 1, true);

                coords.setCoordinate(rowIndex, L(0));
                coords.setCoordinate(colIndex, L(0));

                auto rowOffset     = MkVGPR(DataType::Int64);
                auto rowOffsetExpr = rowOffset->expression();
                auto offset        = MkVGPR(DataType::Int64);
                auto offsetExpr    = offset->expression();

                auto baseIndexExpr = coords.forward({user})[0];
                co_yield generate(rowOffset, baseIndexExpr * L(numBytes));

                // row stride
                auto rowStride     = MkVGPR(DataType::Int64);
                auto rowStrideExpr = rowStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.forwardStride(rowIndex, L(1), {user}, simplify)[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(rowStride, sgpr);
                }

                // col stride
                auto colStride     = MkVGPR(DataType::Int64);
                auto colStrideExpr = colStride->expression();
                {
                    auto sgpr = MkSGPR(DataType::Int64);
                    auto expr = coords.forwardStride(colIndex, L(1), {user}, simplify)[0];
                    co_yield generate(sgpr, expr * L(numBytes));
                    co_yield copy(colStride, sgpr);
                }

                // TODO multidimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(offset, rowOffset);
                    for(int j = 0; j < n; ++j)
                    {
                        co_yield m_context->mem()->store(
                            MemoryInstructions::MemoryKind::Flat,
                            v_ptr,
                            vgpr->element({static_cast<int>(i * n + j)}),
                            offset,
                            numBytes);

                        if(j < n - 1)
                            co_yield generate(offset, offsetExpr + colStrideExpr);
                    }
                    if(i < m - 1)
                        co_yield generate(rowOffset, rowOffsetExpr + rowStrideExpr);
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

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                auto v_ptr = s_ptr->placeholder(Register::Type::Vector);
                co_yield v_ptr->allocate();

                co_yield m_context->copier()->copy(v_ptr, s_ptr);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);
                co_yield offset->allocate();

                auto value = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, agpr->variableType(), 1);
                co_yield value->allocate();

                auto numBytes = DataTypeInfo::Get(agpr->variableType()).elementSize;

                for(uint a = 0; a < num_vgpr; ++a)
                {
                    coords.setCoordinate(VGPR(tile.tag, true), literal(a));

                    auto user_indexes = coords.forward({user});
                    auto user_index
                        = fastDivision(fastMultiplication(simplify(user_indexes[0])), m_context);
                    co_yield generateOffset(offset, user_index, agpr->variableType().dataType);
                    co_yield m_context->copier()->copy(value, agpr->element({static_cast<int>(a)}));

                    co_yield m_context->mem()->store(
                        MemoryInstructions::MemoryKind::Flat, v_ptr, value, offset, numBytes);
                }
            }

            Generator<Instruction> operator()(ControlGraph::StoreTiled const& store,
                                              Transformer                     coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreTiled()");
                co_yield_(Instruction::Comment("GEN: StoreTiled"));

                switch(store.tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(store, store.user, store.tile, coords);
                    break;
                case MemoryType::WAVE:
                {
                    auto waveTile = m_graph.coordinates.getDimension(WaveTile(store.tag, 2, true));
                    co_yield storeMacroTileWAVE(store, store.user, waveTile, coords);
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

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(store.tile.tag));

                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                auto ldsAllocation = Register::Value::AllocateLDS(
                    m_context, vtype, product(store.tile.subTileSizes) * product(m_workgroupSize));

                m_context->registerTagManager()->addRegister(getTag(store.lds).ctag, ldsAllocation);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                coords.setCoordinate(Workitem(store.lds.tag, 0, nullptr, true), m_workitem[0]);
                coords.setCoordinate(Workitem(store.lds.tag, 1, nullptr, true), m_workitem[1]);

                auto lds_offset_expr
                    = (Expression::literal(ldsAllocation->getLDSAllocation()->offset())
                       + coords.forward({store.lds})[0])
                      * Expression::literal(product(store.tile.subTileSizes));

                auto numBytes = DataTypeInfo::Get(vtype).elementSize;
                co_yield generate(lds_offset,
                                  Expression::simplify(Expression::fuse(
                                      m_fastArith(lds_offset_expr * L(numBytes)))));

                // Temporary register that is used to copy the data from global memory to
                // local memory.
                auto vgpr = m_context->registerTagManager()->getRegister(getTag(store.tile).ctag);

                auto const m = store.tile.subTileSizes[0];
                auto const n = store.tile.subTileSizes[1];

                // TODO multi dimensional tiles
                for(int i = 0; i < m; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        // Store the value into local memory
                        co_yield m_context->mem()->storeLocal(
                            lds_offset->subset({0}),
                            vgpr->element({i * n + j}),
                            std::to_string((i * n + j) * numBytes),
                            numBytes);
                    }
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

                auto indexes = coords.forward({store.user});

                co_yield offset->allocate();
                co_yield generateOffset(offset, indexes[0], src->variableType().dataType);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(store.user.argumentName(), s_ptr);

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

        Generator<Instruction> generate(KernelGraph                     graph,
                                        std::shared_ptr<Command>        command,
                                        std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::generate");
            rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}", graph.toDOT());

            auto visitor = CFCodeGeneratorVisitor(graph, command, kernel);

            co_yield visitor.generate();
        }
    }
}
