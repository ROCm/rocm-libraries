
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
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
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
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        /*
         * Code generation
         */
        struct CodeGeneratorVisitor
        {
            CodeGeneratorVisitor(KernelGraph graph, std::shared_ptr<AssemblyKernel> kernel)
                : m_graph(graph)
                , m_kernel(kernel)
                , m_context(kernel->context())
                , m_fastArith{kernel->context()}
            {
            }

            Generator<Instruction> generate()
            {
                auto coords = Transformer(
                    std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                        m_graph.coordinates),
                    m_context,
                    m_fastArith);

                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() begin");
                co_yield setup();
                auto candidates = m_graph.control.roots().to<std::set>();
                co_yield generate(candidates, coords);
                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() end");
            }

            inline Register::ValuePtr MkVGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Vector, type, count);
            }

            inline Register::ValuePtr MkSGPR(VariableType const& type, int count = 1) const
            {
                return Register::Value::Placeholder(m_context, Register::Type::Scalar, type, count);
            }

            inline Register::ValuePtr getBufferSrd(int tag)
            {
                auto bufferTag = m_graph.mapper.get<Buffer>(tag);
                return m_context->registerTagManager()->getRegister(bufferTag);
            }

            Register::ValuePtr getOffset(int tag, int dimension)
            {
                Register::ValuePtr offset;

                auto offsetTag = m_graph.mapper.get<Offset>(tag, dimension);
                if(offsetTag >= 0)
                    offset = m_context->registerTagManager()->getRegister(offsetTag);

                return offset;
            }

            Generator<Instruction>
                generateStride(Register::ValuePtr& stride, int tag, int dimension)
            {
                auto strideTag = m_graph.mapper.get<Stride>(tag, dimension);
                if(strideTag >= 0)
                {
                    auto [strideExpr, strideDataType]
                        = m_context->registerTagManager()->getExpression(strideTag);
                    strideExpr = m_fastArith(strideExpr);

                    // If stride can be evaluated at compile time, return a literal. Otherwise,
                    // create a new register.
                    if(Expression::evaluationTimes(strideExpr)[EvaluationTime::Translate])
                    {
                        stride = Register::Value::Literal(Expression::evaluate(strideExpr));
                    }
                    else
                    {
                        stride = Register::Value::Placeholder(
                            m_context, Register::Type::Scalar, strideDataType, 1);
                        co_yield generate(stride, strideExpr);
                    }
                }
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
                auto inputTags = m_graph.control.getInputNodeIndices<Sequence>(tag);
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
            Generator<Instruction> generate(std::set<int> candidates, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    concatenate("KernelGraph::CodeGenerator::generate: ", candidates));

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
                        auto op = std::get<Operation>(m_graph.control.getElement(tag));
                        co_yield call(tag, op, coords);
                    }

                    // Add output nodes to candidates.

                    for(auto const& tag : nodes)
                    {
                        auto outTags = m_graph.control.getOutputNodeIndices<Sequence>(tag);
                        candidates.insert(outTags.begin(), outTags.end());
                    }

                    // Delete generated nodes from candidates.

                    for(auto const& node : nodes)
                        candidates.erase(node);
                }
            }

            Generator<Instruction>
                call(int tag, Operation const& operation, Transformer const& coords)
            {
                auto opName = toString(operation);
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::{}({})", opName, tag);
                co_yield Instruction::Comment(concatenate(opName, "(", tag, ") BEGIN"));

                AssertFatal(m_completedControlNodes.find(tag) == m_completedControlNodes.end(),
                            ShowValue(operation));

                co_yield std::visit(
                    *this, std::variant<int>(tag), operation, std::variant<Transformer>(coords));

                co_yield Instruction::Comment(concatenate(opName, "(", tag, ") END"));

                m_completedControlNodes.insert(tag);
            }

            Generator<Instruction> operator()(int tag, Kernel const& edge, Transformer coords)
            {
                auto scope = std::make_shared<ScopeManager>(m_context);
                m_context->setScopeManager(scope);

                scope->pushNewScope();
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
                scope->popAndReleaseScope();

                m_context->setScopeManager(nullptr);
            }

            Generator<Instruction> operator()(int tag, Scope const&, Transformer coords)
            {
                auto scope = m_context->getScopeManager();
                scope->pushNewScope();
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
                scope->popAndReleaseScope();
            }

            Generator<Instruction> operator()(int tag, ForLoopOp const& op, Transformer coords)
            {
                auto topLabel = m_context->labelAllocator()->label("ForLoopTop");
                auto botLabel = m_context->labelAllocator()->label("ForLoopBottom");

                co_yield Instruction::Comment("Initialize For Loop");
                auto init = m_graph.control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                co_yield generate(init, coords);

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1);
                auto loop_incr_tag = connections[0].coordinate;
                auto iterReg       = m_context->registerTagManager()->getRegister(loop_incr_tag);
                {
                    auto loopDims
                        = m_graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loop_incr_tag);
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

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);

                co_yield Instruction::Comment("For Loop Increment");
                auto incr
                    = m_graph.control.getOutputNodeIndices<ForLoopIncrement>(tag).to<std::set>();
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
                co_yield Instruction::Unlock("Unlock For Loop");
            }

            Generator<Instruction> operator()(int tag, UnrollOp const& unroll, Transformer coords)
            {
                Throw<FatalError>("Not implemented yet.");
            }

            Generator<Instruction> operator()(int tag, Assign const& assign, Transformer coords)
            {
                auto dim_tag = m_graph.mapper.get(tag, NaryArgument::DEST);

                rocRoller::Log::getLogger()->debug("  assigning dimension: {}", dim_tag);

                auto scope = m_context->getScopeManager();
                scope->addRegister(dim_tag);

                auto deferred = resultVariableType(assign.expression).dataType == DataType::None
                                && !m_context->registerTagManager()->hasRegister(dim_tag);

                Register::ValuePtr dest;
                if(!deferred)
                {
                    rocRoller::Log::getLogger()->debug("  immediate: count {}", assign.valueCount);
                    dest = m_context->registerTagManager()->getRegister(
                        dim_tag,
                        assign.regType,
                        resultVariableType(assign.expression),
                        assign.valueCount);
                }
                co_yield Expression::generate(dest, assign.expression, m_context);

                if(deferred)
                {
                    m_context->registerTagManager()->addRegister(dim_tag, dest);
                }
            }

            Generator<Instruction>
                operator()(int tag, Deallocate const& deallocate, Transformer coords)
            {
                auto dim_tag = m_graph.mapper.get<Dimension>(tag);
                rocRoller::Log::getLogger()->debug("  deallocate dimension: {}", dim_tag);
                m_context->registerTagManager()->deleteTag(dim_tag);
                co_return;
            }

            Generator<Instruction> operator()(int, Barrier const&, Transformer)
            {
                co_yield m_context->mem()->barrier();
            }

            Generator<Instruction> operator()(int tag, ComputeIndex const& ci, Transformer coords)
            {
                auto base = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
                auto offset = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
                auto stride = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
                auto target = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::TARGET});
                auto increment = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::INCREMENT});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::ComputeIndex({}): target {} offset {} stride {}",
                    tag,
                    target,
                    offset,
                    stride);

                auto scope    = m_context->getScopeManager();
                uint numBytes = DataTypeInfo::Get(ci.valueType).elementSize;

                coords.setCoordinate(increment, L(0u));
                for(int idx = 0;; ++idx)
                {
                    auto zeroTag = m_graph.mapper.get(
                        tag,
                        Connections::ComputeIndex{Connections::ComputeIndexArgument::ZERO, idx});
                    if(zeroTag < 0)
                        break;
                    coords.setCoordinate(zeroTag, L(0u));
                }

                auto offsetReg = m_context->registerTagManager()->getRegister(
                    offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("offset", tag));
                co_yield Register::AllocateIfNeeded(offsetReg);
                scope->addRegister(offset);

                if(base < 0)
                {
                    auto indexExpr
                        = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Offset({}): {}", offset, toString(indexExpr));
                    co_yield generate(offsetReg, indexExpr * L(numBytes));
                }

                if(stride > 0)
                {
                    auto indexExpr = ci.forward
                                         ? coords.forwardStride(increment, L(1), {target})[0]
                                         : coords.reverseStride(increment, L(1), {target})[0];
                    rocRoller::Log::getLogger()->debug(
                        "  Stride({}): {}", stride, toString(indexExpr));
                    m_context->registerTagManager()->addExpression(
                        stride, indexExpr * L(numBytes), ci.strideType);
                    scope->addRegister(stride);
                }

                auto buffer = m_graph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});
                if(buffer > 0)
                {
                    auto user = m_graph.coordinates.get<User>(target);
                    if(user)
                    {
                        auto bufferReg = m_context->registerTagManager()->getRegister(
                            buffer,
                            Register::Type::Scalar,
                            {DataType::None, PointerType::Buffer},
                            1);
                        bufferReg->setName(concatenate("buffer", tag));
                        if(bufferReg->allocationState() == Register::AllocationState::Unallocated)
                        {
                            co_yield Register::AllocateIfNeeded(bufferReg);
                            auto basePointer = MkSGPR(DataType::Int64);
                            auto bufDesc     = BufferDescriptor(bufferReg, m_context);
                            co_yield m_context->argLoader()->getValue(user->argumentName(),
                                                                      basePointer);
                            co_yield bufDesc.setBasePointer(basePointer);
                            co_yield bufDesc.setDefaultOpts();
                        }
                        scope->addRegister(buffer);
                    }
                }
            }

            Generator<Instruction>
                operator()(int tag, SetCoordinate const& setCoordinate, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::SetCoordinate({}): {}",
                    tag,
                    Expression::toString(setCoordinate.value));

                auto connections = m_graph.mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid SetCoordinate operation; coordinate missing.");
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
            }

            Generator<Instruction> operator()(int tag, LoadLinear const& edge, Transformer coords)
            {
                Throw<FatalError>("LoadLinear present in kernel graph.");
            }

            /**
             * @brief Load a tile from memory into registers
             *
             * @param kind The kind of memory instruction to use
             * @param m Number of rows in the tile
             * @param n Number of columns in the tile
             * @param dataType The type of the data being loaded
             * @param pack Whether to pack smaller types into a single register
             * @param tag The tag of the control graph node generating the load
             * @param vgpr The registers to store the data in
             * @param offset Offset from the starting index
             * @return Generator<Instruction>
             */
            Generator<Instruction> loadTile(MemoryInstructions::MemoryKind kind,
                                            uint64_t                       m,
                                            uint64_t                       n,
                                            VariableType                   dataType,
                                            bool                           pack,
                                            int                            tag,
                                            Register::ValuePtr             vgpr,
                                            Register::ValuePtr             offset)
            {
                // Get the values from the associated ComputeIndex node
                auto row_offset_reg = getOffset(tag, 0);
                auto col_offset_reg = getOffset(tag, 1);

                std::shared_ptr<BufferDescriptor> bufDesc;
                if(kind == MemoryInstructions::MemoryKind::Buffer)
                {
                    auto bufferSrd = getBufferSrd(tag);
                    bufDesc        = std::make_shared<BufferDescriptor>(bufferSrd, m_context);
                }

                Register::ValuePtr row_stride_reg, col_stride_reg;
                if(m > 1)
                    co_yield generateStride(row_stride_reg, tag, 0);
                co_yield generateStride(col_stride_reg, tag, 1);

                // Load a tile of Half precision values where each register will hold
                // two half precision values.
                if(pack && vgpr->variableType() == DataType::Halfx2)
                {
                    auto offset1 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, col_offset_reg->variableType(), 1);
                    auto offset2 = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, col_offset_reg->variableType(), 1);

                    for(uint i = 0; i < m; ++i)
                    {
                        co_yield copy(col_offset_reg, row_offset_reg);
                        for(uint j = 0; j < n; j += 2)
                        {
                            uint a = i * n + j;

                            co_yield copy(offset1, col_offset_reg);
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());
                            co_yield copy(offset2, col_offset_reg);
                            co_yield generate(col_offset_reg,
                                              col_offset_reg->expression()
                                                  + col_stride_reg->expression());

                            co_yield m_context->mem()->loadAndPack(
                                kind,
                                vgpr->element({static_cast<int>(a / 2)}),
                                offset1,
                                offset,
                                offset2,
                                offset,
                                "",
                                bufDesc);
                        }
                        if(i < m - 1)
                            co_yield generate(row_offset_reg,
                                              row_offset_reg->expression()
                                                  + row_stride_reg->expression());
                    }
                }
                else
                {
                    auto elementSize = (uint)DataTypeInfo::Get(dataType).elementSize;
                    for(int i = 0; i < m; ++i)
                    {
                        co_yield copy(col_offset_reg, row_offset_reg);

                        for(int j = 0; j < n; ++j)
                        {
                            co_yield m_context->mem()->load(
                                kind,
                                vgpr->element({static_cast<int>(i * n + j)}),
                                col_offset_reg->subset({0}),
                                offset,
                                elementSize,
                                "",
                                false,
                                bufDesc);
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
            }

            Generator<Instruction>
                loadMacroTileVGPRCI(int tag, LoadTiled const& load, Transformer coords, int sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileVGPRCI()");
                co_yield Instruction::Comment("GEN: loadMacroTileVGPRCI");

                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto mac_offset_reg = getOffset(tag, -1);
                auto row_offset_reg = getOffset(tag, 0);

                std::shared_ptr<Register::Value> tmpl;
                if(load.vtype == DataType::Half)
                    tmpl = MkVGPR(DataType::Halfx2, product(mac_tile.subTileSizes));
                else
                    tmpl = MkVGPR(load.vtype, product(mac_tile.subTileSizes));

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                co_yield copy(row_offset_reg, mac_offset_reg);

                co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                                  m,
                                  n,
                                  load.vtype,
                                  false,
                                  tag,
                                  vgpr,
                                  nullptr);
            }

            Generator<Instruction>
                loadMacroTileVGPR(int tag, LoadTiled const& load, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileVGPR({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileVGPR");

                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto tmpl = MkVGPR(load.vtype, product(mac_tile.subTileSizes));
                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

                rocRoller::Log::getLogger()->debug(
                    "  macro tile: {}; sub tile size: {}x{}", mac_tile_tag, m, n);

                co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                                  m,
                                  n,
                                  vgpr->variableType(),
                                  false,
                                  tag,
                                  vgpr,
                                  nullptr);
            }

            Generator<Instruction>
                loadMacroTileLDS(int tag, LoadLDSTile const& load, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileLDS()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileLDS"));

                auto [lds_tag, lds]   = m_graph.getDimension<LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<MacroTile>(tag);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                std::shared_ptr<Register::Value> tmpl;
                if(load.vtype == DataType::Half)
                    tmpl = MkVGPR(DataType::Halfx2, product(tile.subTileSizes));
                else
                    tmpl = MkVGPR(load.vtype, product(tile.subTileSizes));

                auto vgpr = m_context->registerTagManager()->getRegister(tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                co_yield loadTile(MemoryInstructions::MemoryKind::Local,
                                  m,
                                  n,
                                  load.vtype,
                                  false,
                                  tag,
                                  vgpr,
                                  lds_offset);
            }

            Generator<Instruction> loadMacroTileWAVELDSCI(int                tag,
                                                          LoadLDSTile const& load,
                                                          Transformer        coords,
                                                          int                sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVELDSCI()");
                co_yield_(Instruction::Comment("GEN: loadMacroTileWAVELDSCI"));

                auto [lds_tag, lds]             = m_graph.getDimension<LDS>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);

                // Find the LDS allocation that contains the tile and store
                // the offset of the beginning of the allocation into lds_offset.
                auto ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                auto vtype = ldsAllocation->variableType();

                auto n_wave_tag = m_graph.mapper.get<WaveTileNumber>(tag, sdim);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                Register::ValuePtr tmpl;
                if(load.vtype == DataType::Half)
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);
                }
                else
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);
                }

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                co_yield loadTile(MemoryInstructions::MemoryKind::Local,
                                  1,
                                  num_vgpr,
                                  load.vtype,
                                  true,
                                  tag,
                                  vgpr,
                                  lds_offset);
            }

            // CI : compute index
            Generator<Instruction>
                loadMacroTileWAVECI(int tag, LoadTiled const& load, Transformer coords, int sdim)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECI({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECI");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);

                Register::ValuePtr basePointer;
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                Register::ValuePtr tmpl;
                if(load.vtype == DataType::Half)
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);
                }
                else
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);
                }

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                                  1,
                                  num_vgpr,
                                  load.vtype,
                                  true,
                                  tag,
                                  vgpr,
                                  nullptr);
            }

            Generator<Instruction>
                loadMacroTileWAVECIACCUM(int tag, LoadTiled const& load, Transformer coords)

            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::loadMacroTileWAVECIACCUM({})", tag);
                co_yield Instruction::Comment("GEN: loadMacroTileWAVECIACCUM");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);
                auto mac_tile_tag               = m_graph.mapper.get<MacroTile>(tag);

                // Move the argument pointer into v_ptr
                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                Register::ValuePtr tmpl;
                if(load.vtype == DataType::Half)
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, DataType::Halfx2, num_vgpr / 2);
                }
                else
                {
                    tmpl = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, load.vtype, num_vgpr);
                }

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag, tmpl);
                co_yield Register::AllocateIfNeeded(vgpr);

                co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                                  num_vgpr / 4,
                                  4,
                                  load.vtype,
                                  true,
                                  tag,
                                  vgpr,
                                  nullptr);
            }

            Generator<Instruction> operator()(int tag, LoadTiled const& load, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileVGPRCI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileVGPRCI(tag, load, coords, 0);
                        break;
                    default:
                        co_yield loadMacroTileVGPR(tag, load, coords);
                        break;
                    }
                }
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
                        Throw<FatalError>("Layout type not supported yet for LoadTiled.");
                    }
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet for LoadTiled.");
                }
            }

            Generator<Instruction> operator()(int tag, LoadLDSTile const& load, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield loadMacroTileLDS(tag, load, coords);
                    break;
                case MemoryType::WAVE:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_A:
                        co_yield loadMacroTileWAVELDSCI(tag, load, coords, 1);
                        break;
                    case LayoutType::MATRIX_B:
                        co_yield loadMacroTileWAVELDSCI(tag, load, coords, 0);
                        break;
                    default:
                        Throw<FatalError>("Layout type not supported yet for LoadLDSTile.");
                    }
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet for LoadLDSTile.");
                }
            }

            Generator<Instruction> operator()(int tag, LoadVGPR const& load, Transformer coords)
            {
                auto [userTag, user] = m_graph.getDimension<User>(tag);
                auto [vgprTag, vgpr] = m_graph.getDimension<VGPR>(tag);

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

            Generator<Instruction> loadVGPRFromScalarValue(User                             user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           Transformer                      coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar value");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar value");

                Register::ValuePtr s_value;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_value);
                co_yield m_context->copier()->copy(vgpr, s_value, "Move value");
            }

            Generator<Instruction> loadVGPRFromScalarPointer(User                             user,
                                                             std::shared_ptr<Register::Value> vgpr,
                                                             Transformer coords)
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
                                                           User                             user,
                                                           std::shared_ptr<Register::Value> vgpr,
                                                           Transformer                      coords)
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

            Generator<Instruction> operator()(int tag, Multiply const& mult, Transformer coords)
            {
                auto loads = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::vector>();
                AssertFatal(loads.size() == 2, "Multiply op needs two operands");

                auto loadA = m_graph.control.getElement(loads[0]);
                auto loadB = m_graph.control.getElement(loads[1]);

                int sourceA_tag = -1;
                if(isOperation<LoadTiled>(loadA))
                    sourceA_tag = m_graph.mapper.get<User>(tag, 0);
                else if(isOperation<LoadLDSTile>(loadA))
                    sourceA_tag = m_graph.mapper.get<LDS>(loads[0]);

                int sourceB_tag = -1;
                if(isOperation<LoadTiled>(loadB))
                    sourceB_tag = m_graph.mapper.get<User>(tag, 1);
                else if(isOperation<LoadLDSTile>(loadB))
                    sourceB_tag = m_graph.mapper.get<LDS>(loads[1]);

                AssertFatal(sourceA_tag > 0 && sourceB_tag > 0, "User or LDS dimensions not found");

                auto [waveA_tag, waveA] = m_graph.getDimension<WaveTile>(
                    tag, Connections::typeArgument<WaveTile>(NaryArgument::LHS));
                auto [waveB_tag, waveB] = m_graph.getDimension<WaveTile>(
                    tag, Connections::typeArgument<WaveTile>(NaryArgument::RHS));

                auto [macA_tag, macA] = m_graph.getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                auto [macB_tag, macB] = m_graph.getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::RHS));

                auto n_waveA_y_tags
                    = m_graph.coordinates
                          .findNodes(sourceA_tag,
                                     [&](int index) -> bool {
                                         auto node = m_graph.coordinates.get<WaveTileNumber>(index);
                                         if(node)
                                             return node->dim == 1;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveA_y_tags.size() == 1);

                auto n_waveB_x_tags
                    = m_graph.coordinates
                          .findNodes(sourceB_tag,
                                     [&](int index) -> bool {
                                         auto node = m_graph.coordinates.get<WaveTileNumber>(index);
                                         if(node)
                                             return node->dim == 0;
                                         return false;
                                     })
                          .to<std::vector>();
                AssertFatal(n_waveB_x_tags.size() == 1);

                auto loadAB = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();

                auto mac_offset_x_reg  = getOffset(loads[0], -1);
                auto wave_offset_x_reg = getOffset(loads[0], 0);

                auto mac_offset_y_reg  = getOffset(loads[1], -1);
                auto wave_offset_y_reg = getOffset(loads[1], 0);

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint num_elements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_agpr     = num_elements / wfs;

                auto [D_tag, _D] = m_graph.getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

                auto D = m_context->registerTagManager()->getRegister(
                    D_tag, Register::Type::Accumulator, DataType::Float, num_agpr);

                auto completed = m_completedControlNodes;

                // D is not initialized here

                if(mac_offset_x_reg)
                    co_yield copy(wave_offset_x_reg, mac_offset_x_reg);
                if(mac_offset_y_reg)
                    co_yield copy(wave_offset_y_reg, mac_offset_y_reg);

                // saving the offsets to be restored for each macrotile in LDS
                // TODO : Need more design thought (how to seed an offset register)
                auto reset_offset_x = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                if(isOperation<LoadLDSTile>(loadA))
                    co_yield copy(reset_offset_x, wave_offset_x_reg);

                auto reset_offset_y = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                if(isOperation<LoadLDSTile>(loadB))
                    co_yield copy(reset_offset_y, wave_offset_y_reg);

                Register::ValuePtr wave_stride_x_reg, wave_stride_y_reg;
                co_yield generateStride(wave_stride_x_reg, loads[0], 0);
                co_yield generateStride(wave_stride_y_reg, loads[1], 0);

                AssertFatal(wave_stride_x_reg, "Invalid WAVE X stride register.");
                AssertFatal(wave_stride_y_reg, "Invalid WAVE Y stride register.");

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
                        std::make_shared<WaveTile>(waveA));
                    Expression::ExpressionPtr B = std::make_shared<Expression::Expression>(
                        std::make_shared<WaveTile>(waveB));

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

                if(isOperation<LoadLDSTile>(loadA))
                    co_yield copy(wave_offset_x_reg, reset_offset_x);
                if(isOperation<LoadLDSTile>(loadB))
                    co_yield copy(wave_offset_y_reg, reset_offset_y);
            }

            Generator<Instruction>
                operator()(int tag, TensorContraction const& mul, Transformer coords)
            {
                Throw<FatalError>("TensorContraction present in kernel graph.");
            }

            Generator<Instruction> operator()(int tag, StoreLinear const& edge, Transformer coords)
            {
                Throw<FatalError>("StoreLinear present in kernel graph.");
            }

            /**
             * @brief Store a tile from registers into memory
             *
             * @param kind The kind of memory instruction to use
             * @param m Number of rows in the tile
             * @param n Number of columns in the tile
             * @param dataType The type of the data being stored
             * @param tag The tag of the control graph node generating the store
             * @param vgpr The registers containing the data
             * @param offset Offset from the starting index
             * @return Generator<Instruction>
             */
            Generator<Instruction> storeTile(MemoryInstructions::MemoryKind kind,
                                             uint64_t                       m,
                                             uint64_t                       n,
                                             VariableType                   dataType,
                                             int                            tag,
                                             Register::ValuePtr             vgpr,
                                             Register::ValuePtr             offset)
            {
                auto elementSize    = DataTypeInfo::Get(dataType).elementSize;
                auto row_offset_reg = getOffset(tag, 0);
                auto col_offset_reg = getOffset(tag, 1);

                std::shared_ptr<BufferDescriptor> bufDesc;
                if(kind == MemoryInstructions::MemoryKind::Buffer)
                {
                    auto bufferSrd = getBufferSrd(tag);
                    bufDesc        = std::make_shared<BufferDescriptor>(bufferSrd, m_context);
                }

                Register::ValuePtr row_stride_reg, col_stride_reg;
                co_yield generateStride(row_stride_reg, tag, 0);
                co_yield generateStride(col_stride_reg, tag, 1);

                for(int i = 0; i < m; ++i)
                {
                    co_yield copy(col_offset_reg, row_offset_reg);

                    for(int j = 0; j < n; ++j)
                    {
                        uint a = i * n + j;

                        Register::ValuePtr value;
                        if(vgpr->regType() == Register::Type::Accumulator)
                        {
                            value = MkVGPR(vgpr->variableType());
                            co_yield m_context->copier()->copy(
                                value, vgpr->element({static_cast<int>(a)}));
                        }
                        else
                        {
                            value = vgpr->element({static_cast<int>(a)});
                        }

                        Register::ValuePtr converted;
                        if(DataTypeInfo::Get(value->variableType()).segmentVariableType != dataType)
                        {
                            converted = MkVGPR(dataType);
                            co_yield Expression::generate(
                                converted,
                                convert(dataType.dataType,
                                        std::make_shared<Expression::Expression>(value)),
                                m_context);
                        }
                        else
                        {
                            converted = value;
                        }

                        co_yield m_context->mem()->store(kind,
                                                         col_offset_reg->subset({0}),
                                                         converted,
                                                         offset,
                                                         elementSize,
                                                         "",
                                                         bufDesc);
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

            Generator<Instruction>
                storeMacroTileLDS(int tag, StoreLDSTile const& store, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileLDS()");
                co_yield_(Instruction::Comment("GEN: storeMacroTileLDS"));

                auto [lds_tag, lds]   = m_graph.getDimension<LDS>(tag);
                auto [tile_tag, tile] = m_graph.getDimension<MacroTile>(tag);

                // Temporary register(s) that is used to copy the data from global memory to
                // local memory.
                auto vgpr  = m_context->registerTagManager()->getRegister(tile_tag);
                auto vtype = store.dataType;

                auto row_offset_reg = getOffset(tag, 0);

                auto numElements = product(tile.subTileSizes) * product(m_workgroupSize);
                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                Register::ValuePtr ldsAllocation;
                if(!m_context->registerTagManager()->hasRegister(lds_tag))
                {
                    ldsAllocation = Register::Value::AllocateLDS(m_context, vtype, numElements);
                    m_context->registerTagManager()->addRegister(lds_tag, ldsAllocation);
                }
                else
                {
                    ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);
                }

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                auto lds_offset_expr
                    = Expression::literal(ldsAllocation->getLDSAllocation()->offset());
                co_yield generate(lds_offset, lds_offset_expr);

                auto const m = tile.subTileSizes[0];
                auto const n = tile.subTileSizes[1];

                // saving the offsets to be restored for each macrotile in LDS
                // TODO : Need more design thought (how to seed an offset register)
                auto reset_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield copy(reset_offset, row_offset_reg);

                co_yield storeTile(
                    MemoryInstructions::MemoryKind::Local, m, n, vtype, tag, vgpr, lds_offset);

                co_yield copy(row_offset_reg, reset_offset);
            }

            Generator<Instruction>
                storeMacroTileVGPR(int tag, StoreTiled const& store, Transformer coords)
            {
                auto [user_tag, user]         = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileVGPR({})", tag);
                co_yield Instruction::Comment("GEN: storeMacroTileVGPR");

                rocRoller::Log::getLogger()->debug("  user {}; tile {}", user_tag, mac_tile_tag);

                auto vgpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                auto basePointer = MkSGPR(DataType::Int64);
                co_yield m_context->argLoader()->getValue(user.argumentName(), basePointer);

                auto const m = mac_tile.subTileSizes[0];
                auto const n = mac_tile.subTileSizes[1];

                co_yield storeTile(MemoryInstructions::MemoryKind::Buffer,
                                   m,
                                   n,
                                   store.dataType,
                                   tag,
                                   vgpr,
                                   nullptr);
            }

            Generator<Instruction>
                storeMacroTileWAVELDS(int tag, StoreLDSTile const& store, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVELDS()");
                co_yield_(Instruction::Comment("GEN: storeMacroTileWAVELDS"));

                auto [lds_tag, lds]             = m_graph.getDimension<LDS>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);
                auto macrotileNumElements       = product(mac_tile.sizes);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);
                uint wavetileNumElements        = wave_tile.sizes[0] * wave_tile.sizes[1];
                auto vtype                      = store.dataType;

                // Allocate LDS memory, and store the offset of the beginning of the allocation
                // into lds_offset.
                Register::ValuePtr ldsAllocation;
                if(!m_context->registerTagManager()->hasRegister(lds_tag))
                {
                    ldsAllocation
                        = Register::Value::AllocateLDS(m_context, vtype, macrotileNumElements);
                    m_context->registerTagManager()->addRegister(lds_tag, ldsAllocation);
                }
                else
                {
                    ldsAllocation = m_context->registerTagManager()->getRegister(lds_tag);
                }

                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);
                co_yield generate(lds_offset,
                                  Expression::literal(ldsAllocation->getLDSAllocation()->offset()));

                uint wfs      = m_context->kernel()->wavefront_size();
                uint num_vgpr = wavetileNumElements / wfs;
                auto agpr     = m_context->registerTagManager()->getRegister(mac_tile_tag);
                AssertFatal(agpr->registerCount() == num_vgpr);

                co_yield storeTile(MemoryInstructions::MemoryKind::Local,
                                   num_vgpr / 4,
                                   4,
                                   vtype,
                                   tag,
                                   agpr,
                                   lds_offset);
            }

            Generator<Instruction>
                storeMacroTileWAVECI(int tag, StoreTiled const& store, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::storeMacroTileWAVE()");
                co_yield Instruction::Comment("GEN: storeMacroTileWAVE");

                auto [user_tag, user]           = m_graph.getDimension<User>(tag);
                auto [mac_tile_tag, mac_tile]   = m_graph.getDimension<MacroTile>(tag);
                auto [wave_tile_tag, wave_tile] = m_graph.getDimension<WaveTile>(tag);

                uint num_elements = wave_tile.sizes[0] * wave_tile.sizes[1];
                uint wfs          = m_context->kernel()->wavefront_size();
                uint num_vgpr     = num_elements / wfs;

                auto agpr = m_context->registerTagManager()->getRegister(mac_tile_tag);

                AssertFatal(agpr->registerCount() == num_vgpr);

                Register::ValuePtr s_ptr;
                co_yield m_context->argLoader()->getValue(user.argumentName(), s_ptr);

                co_yield storeTile(MemoryInstructions::MemoryKind::Buffer,
                                   num_vgpr / 4,
                                   4,
                                   store.dataType,
                                   tag,
                                   agpr,
                                   nullptr);
            }

            Generator<Instruction> operator()(int tag, StoreTiled const& store, Transformer coords)
            {
                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                    co_yield storeMacroTileVGPR(tag, store, coords);
                    break;
                case MemoryType::WAVE:
                    co_yield storeMacroTileWAVECI(tag, store, coords);
                    break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet for StoreTiled.");
                }
            }

            Generator<Instruction>
                operator()(int tag, StoreLDSTile const& store, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreLDSTiled({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: StoreLDSTile");

                auto [mac_tile_tag, mac_tile] = m_graph.getDimension<MacroTile>(tag);

                switch(mac_tile.memoryType)
                {
                case MemoryType::VGPR:
                case MemoryType::LDS:
                    co_yield storeMacroTileLDS(tag, store, coords);
                    break;
                case MemoryType::WAVE:
                {
                    switch(mac_tile.layoutType)
                    {
                    case LayoutType::MATRIX_ACCUMULATOR:
                        co_yield storeMacroTileWAVELDS(tag, store, coords);
                        break;
                    default:
                        Throw<FatalError>("Layout type not supported yet for StoreLDSTile.");
                    }
                }
                break;
                default:
                    Throw<FatalError>("Tile affinity type not supported yet for StoreLDSTile.");
                }
            }

            Generator<Instruction> operator()(int tag, StoreVGPR const& store, Transformer coords)
            {
                co_yield Instruction::Comment("GEN: StoreVGPR");

                auto [vgprTag, vgpr] = m_graph.getDimension<VGPR>(tag);
                auto [userTag, user] = m_graph.getDimension<User>(tag);

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
            KernelGraph                     m_graph;
            std::shared_ptr<Context>        m_context;
            std::shared_ptr<AssemblyKernel> m_kernel;

            std::set<int> m_completedControlNodes;

            std::vector<ExpressionPtr> m_workgroup;
            std::vector<ExpressionPtr> m_workitem;
            std::vector<unsigned int>  m_workgroupSize;
            FastArithmetic             m_fastArith;
        };

        Generator<Instruction> generate(KernelGraph graph, std::shared_ptr<AssemblyKernel> kernel)
        {
            TIMER(t, "KernelGraph::generate");
            rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}",
                                               graph.toDOT(true));

            auto visitor = CodeGeneratorVisitor(graph, kernel);

            co_yield visitor.generate();
        }
    }
}
