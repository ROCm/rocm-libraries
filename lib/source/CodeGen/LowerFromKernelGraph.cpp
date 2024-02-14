
#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
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
            CodeGeneratorVisitor(KernelGraphPtr graph, AssemblyKernelPtr kernel)
                : m_graph(graph)
                , m_kernel(kernel)
                , m_context(kernel->context())
                , m_fastArith{kernel->context()}
                , m_loadStoreTileGenerator(
                      m_graph, kernel->context(), kernel->max_flat_workgroup_size())
            {
            }

            Generator<Instruction> generate()
            {
                m_kernel->startCodeGeneration();

                auto coords = Transformer(
                    std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                        m_graph->coordinates),
                    m_context,
                    m_fastArith);

                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() begin");
                auto candidates = m_graph->control.roots().to<std::set>();
                AssertFatal(candidates.size() == 1,
                            "The control graph should only contain one root node, the Kernel node.",
                            ShowValue(candidates.size()));

                co_yield generate(candidates, coords);
                co_yield Instruction::Comment("CodeGeneratorVisitor::generate() end");
            }

            /**
             * Generate an index from `expr` and store in `dst`
             * register.  Destination register should be an Int64.
             */
            Generator<Instruction> generateOffset(Register::ValuePtr&       dst,
                                                  Expression::ExpressionPtr expr,
                                                  DataType                  dtype,
                                                  Expression::ExpressionPtr offsetInBytes)
            {
                auto const& info     = DataTypeInfo::Get(dtype);
                auto        numBytes = Expression::literal(static_cast<uint>(info.elementSize));

                // TODO: Consider moving numBytes into input of this function.
                if(offsetInBytes)
                    co_yield Expression::generate(
                        dst, m_fastArith(expr * numBytes + offsetInBytes), m_context);
                else
                    co_yield Expression::generate(dst, m_fastArith(expr * numBytes), m_context);
            }

            bool hasGeneratedInputs(int const& tag)
            {
                auto inputTags = m_graph->control.getInputNodeIndices<Sequence>(tag);
                for(auto const& itag : inputTags)
                {
                    if(m_completedControlNodes.find(itag) == m_completedControlNodes.end())
                        return false;
                }
                return true;
            }

            /**
             * Partitions `candidates` into nodes that are ready to be generated and nodes that aren't ready.
             * A node is ready if all the upstream nodes connected via `Sequence` edges have been generated.
             * Nodes that are ready will be removed from `candidates` and will be in the returned set.
             * Nodes that are not ready will remain in `candidates`.
             */
            std::set<int> findAndRemoveSatisfiedNodes(std::set<int>& candidates)
            {
                std::set<int> nodes;

                // Find all candidate nodes whose inputs have been satisfied
                for(auto const& tag : candidates)
                    if(hasGeneratedInputs(tag))
                        nodes.insert(tag);

                // Delete nodes about to be generated from candidates.
                for(auto node : nodes)
                    candidates.erase(node);

                return nodes;
            }

            /**
             * Generate code for the specified nodes and their standard (Sequence) dependencies.
             */
            Generator<Instruction> generate(std::set<int> candidates, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    concatenate("KernelGraph::CodeGenerator::generate: ", candidates));

                auto message = concatenate("generate(", candidates, ")");
                co_yield Instruction::Comment(message);

                candidates = m_graph->control.followEdges<Sequence>(candidates);

                while(!candidates.empty())
                {
                    std::set<int> nodes = findAndRemoveSatisfiedNodes(candidates);

                    // If there are no valid nodes, we have a problem.
                    AssertFatal(!nodes.empty(),
                                "Invalid control graph!",
                                ShowValue(m_graph->control),
                                ShowValue(candidates),
                                ShowValue(m_completedControlNodes));

                    // Generate code for all the nodes we found.

                    std::vector<Generator<Instruction>> generators;
                    for(auto tag : nodes)
                    {
                        auto op = m_graph->control.getNode(tag);
                        generators.push_back(call(tag, op, coords));
                    }

                    if(generators.size() == 1)
                    {
                        co_yield std::move(generators[0]);
                    }
                    else
                    {
                        co_yield Instruction::Comment(
                            concatenate("BEGIN Scheduler for operations ", nodes));
                        auto proc = Settings::getInstance()->get(Settings::Scheduler);
                        auto cost = Settings::getInstance()->get(Settings::SchedulerCost);
                        auto scheduler
                            = Component::GetNew<Scheduling::Scheduler>(proc, cost, m_context);

                        if(!scheduler->supportsAddingStreams())
                        {
                            co_yield (*scheduler)(generators);
                        }
                        else
                        {
                            auto generator         = (*scheduler)(generators);
                            auto numCompletedNodes = m_completedControlNodes.size();

                            for(auto iter = generator.begin(); iter != generator.end(); ++iter)
                            {
                                if(numCompletedNodes != m_completedControlNodes.size()
                                   && !candidates.empty())
                                {
                                    auto newNodes = findAndRemoveSatisfiedNodes(candidates);

                                    if(!newNodes.empty())
                                    {
                                        co_yield Instruction::Comment(
                                            concatenate("ADD operations ",
                                                        newNodes,
                                                        " to scheduler for ",
                                                        nodes));

                                        for(auto tag : newNodes)
                                        {
                                            auto op = m_graph->control.getNode(tag);
                                            generators.push_back(call(tag, op, coords));
                                        }

                                        nodes.insert(newNodes.begin(), newNodes.end());
                                    }

                                    numCompletedNodes = m_completedControlNodes.size();
                                }

                                co_yield *iter;
                            }
                        }

                        co_yield Instruction::Comment(
                            concatenate("END Scheduler for operations ", nodes));
                    }
                }

                co_yield Instruction::Comment("end: " + message);
            }

            /**
             * Note that `operation` must be passed by value (not by reference) to avoid a
             * dangling reference issue if call() is sent into a scheduler instead of being
             * yielded directly.
             */
            Generator<Instruction> call(int tag, Operation operation, Transformer const& coords)
            {
                auto opName = toString(operation);
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::{}({})", opName, tag);
                co_yield Instruction::Comment(concatenate(opName, "(", tag, ") BEGIN"));

                AssertFatal(m_completedControlNodes.find(tag) == m_completedControlNodes.end(),
                            ShowValue(operation),
                            ShowValue(tag));

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
                auto init = m_graph->control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                co_yield generate(init, coords);
                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
                scope->popAndReleaseScope();

                m_context->setScopeManager(nullptr);
            }

            Generator<Instruction> operator()(int tag, Scope const&, Transformer coords)
            {
                auto scope   = m_context->getScopeManager();
                auto message = concatenate("Scope ", tag);

                // Under the current implementation,
                //  - All new DataFlow allocations are associated with the top scope
                //    regardless of if this is correct
                //  - When the scope is popped, all DataFlow registers in that are freed.
                //
                // Until this is changed, we need to lock the scheduler here.

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock " + message);
                scope->pushNewScope();

                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);

                scope->popAndReleaseScope();
                co_yield Instruction::Unlock("Unlock " + message);
            }

            Generator<Instruction> operator()(int tag, ConditionalOp const& op, Transformer coords)
            {
                auto falseLabel = m_context->labelAllocator()->label("ConditionalFalse");
                auto botLabel   = m_context->labelAllocator()->label("ConditionalBottom");

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock for Conditional");
                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(op.condition);
                auto conditionResult = conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();
                co_yield Expression::generate(
                    conditionResult, m_fastArith(op.condition), m_context);

                co_yield m_context->brancher()->branchIfZero(
                    falseLabel,
                    conditionResult,
                    concatenate("Condition: False, jump to ", falseLabel->toString()));
                auto trueBody = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(trueBody, coords);
                co_yield m_context->brancher()->branch(
                    botLabel, concatenate("Condition: Done, jump to ", botLabel->toString()));

                co_yield Instruction::Label(falseLabel);
                auto elseBody = m_graph->control.getOutputNodeIndices<Else>(tag).to<std::set>();
                if(!elseBody.empty())
                {
                    co_yield generate(elseBody, coords);
                }

                co_yield Instruction::Label(botLabel);
                co_yield Instruction::Unlock("Unlock Conditional");
            }

            Generator<Instruction> operator()(int tag, DoWhileOp const& op, Transformer coords)
            {
                auto topLabel = m_context->labelAllocator()->label("DoWhileTop");

                co_yield Instruction::Comment("Initialize DoWhileLoop");

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock DoWhile");

                //Do Body at least once
                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();

                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(op.condition);
                auto conditionResult = conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();

                co_yield Instruction::Label(topLabel);
                co_yield generate(body, coords);

                //Check Condition
                co_yield Expression::generate(
                    conditionResult, m_fastArith(op.condition), m_context);

                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    conditionResult,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                // TODO: Have deallocate nodes generate the proper wait count and remove this wait.
                //       This is currently needed in case there are loads within a loop that are never
                //       used within the loop. If there are, the wait count observer never releases
                //       the registers.
                co_yield Instruction::Wait(
                    WaitCount::Zero("DEBUG: Wait after branch", m_context->targetArchitecture()));
                co_yield Instruction::Unlock("Unlock DoWhile");
            }

            Generator<Instruction> operator()(int tag, ForLoopOp const& op, Transformer coords)
            {
                auto topLabel = m_context->labelAllocator()->label("ForLoopTop");
                auto botLabel = m_context->labelAllocator()->label("ForLoopBottom");

                co_yield Instruction::Comment("Initialize For Loop");
                auto init = m_graph->control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                co_yield generate(init, coords);

                auto loopIncrTag = m_graph->mapper.get(tag, NaryArgument::DEST);

                auto iterReg = m_context->registerTagManager()->getRegister(loopIncrTag);
                {
                    auto loopDims
                        = m_graph->coordinates.getOutputNodeIndices<DataFlowEdge>(loopIncrTag);
                    for(auto const& dim : loopDims)
                    {
                        coords.setCoordinate(dim, iterReg->expression());
                    }
                }

                co_yield Instruction::Lock(Scheduling::Dependency::Branch, "Lock For Loop");
                auto [conditionRegisterType, conditionVariableType]
                    = Expression::resultType(op.condition);
                auto conditionResult = conditionVariableType == DataType::Bool
                                           ? m_context->getSCC()
                                           : m_context->getVCC();
                co_yield Expression::generate(
                    conditionResult, m_fastArith(op.condition), m_context);
                co_yield m_context->brancher()->branchIfZero(
                    botLabel,
                    conditionResult,
                    concatenate("Condition: Top (jump to " + botLabel->toString() + " if false)"));

                co_yield Instruction::Label(topLabel);

                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);

                co_yield Instruction::Comment("For Loop Increment");
                auto incr
                    = m_graph->control.getOutputNodeIndices<ForLoopIncrement>(tag).to<std::set>();
                co_yield generate(incr, coords);
                co_yield Instruction::Comment("Condition: Bottom (jump to " + topLabel->toString()
                                              + " if true)");

                co_yield Expression::generate(
                    conditionResult, m_fastArith(op.condition), m_context);
                co_yield m_context->brancher()->branchIfNonZero(
                    topLabel,
                    conditionResult,
                    concatenate("Condition: Bottom (jump to " + topLabel->toString()
                                + " if true)"));

                co_yield Instruction::Label(botLabel);
                // TODO: Have deallocate nodes generate the proper wait count and remove this wait.
                //       This is currently needed in case there are loads within a loop that are never
                //       used within the loop. If there are, the wait count observer never releases
                //       the registers.
                co_yield Instruction::Wait(
                    WaitCount::Zero("DEBUG: Wait after branch", m_context->targetArchitecture()));
                co_yield Instruction::Unlock("Unlock For Loop");
            }

            Generator<Instruction> operator()(int tag, UnrollOp const& unroll, Transformer coords)
            {
                Throw<FatalError>("Not implemented yet.");
            }

            struct ExpressionHasNoneDTVisitor
            {
                template <CTernary Expr>
                bool operator()(Expr const& expr) const
                {
                    return call(expr.lhs) || call(expr.r1hs) || call(expr.r2hs);
                }

                template <CBinary Expr>
                bool operator()(Expr const& expr) const
                {
                    return call(expr.lhs) || call(expr.rhs);
                }

                template <CUnary Expr>
                bool operator()(Expr const& expr) const
                {
                    return call(expr.arg);
                }

                template <typename T>
                bool operator()(T const& expr) const
                {
                    return false;
                }

                bool operator()(Register::ValuePtr const& expr) const
                {
                    if(!expr)
                        return false;

                    return expr->variableType() == DataType::None;
                }

                bool operator()(DataFlowTag const& expr) const
                {
                    return expr.varType == DataType::None;
                }

                bool call(Expression::Expression const& expr) const
                {
                    return std::visit(*this, expr);
                }

                bool call(ExpressionPtr expr) const
                {
                    if(!expr)
                        return false;

                    return call(*expr);
                }
            };

            /**
             * @brief Returns true if an expression has any values with
             *        a datatype of None.
             *
             * @param expr
             * @return true
             * @return false
             */
            bool expressionHasNoneDT(ExpressionPtr const& expr)
            {
                auto visitor = ExpressionHasNoneDTVisitor();
                return visitor.call(expr);
            }

            Generator<Instruction> operator()(int tag, Assign const& assign, Transformer coords)
            {
                auto dimTag = m_graph->mapper.get(tag, NaryArgument::DEST);

                rocRoller::Log::getLogger()->debug("  assigning dimension: {}", dimTag);
                co_yield Instruction::Comment(
                    concatenate("Assign dim(", dimTag, ") = ", assign.expression));

                auto scope = m_context->getScopeManager();
                scope->addRegister(dimTag);

                auto deferred = expressionHasNoneDT(assign.expression)
                                && !m_context->registerTagManager()->hasRegister(dimTag);

                Register::ValuePtr dest;
                if(!deferred)
                {
                    rocRoller::Log::getLogger()->debug("  immediate: count {}", assign.valueCount);
                    if(assign.regType == Register::Type::Accumulator)
                    {
                        // ACCVGPR should always be contiguous
                        dest = m_context->registerTagManager()->getRegister(
                            dimTag,
                            assign.regType,
                            resultVariableType(assign.expression),
                            assign.valueCount,
                            Register::AllocationOptions{.contiguousChunkWidth
                                                        = static_cast<int>(assign.valueCount)});
                    }
                    else
                    {
                        dest = m_context->registerTagManager()->getRegister(
                            dimTag,
                            assign.regType,
                            resultVariableType(assign.expression),
                            assign.valueCount);
                    }
                    if(dest->name().empty())
                        dest->setName(concatenate("DataFlowTag", dimTag));
                }
                co_yield Expression::generate(dest, m_fastArith(assign.expression), m_context);

                if(deferred)
                {
                    m_context->registerTagManager()->addRegister(dimTag, dest);
                    if(dest->name().empty())
                        dest->setName(concatenate("DataFlowTag", dimTag));
                }
            }

            Generator<Instruction>
                operator()(int tag, Deallocate const& deallocate, Transformer coords)
            {
                auto dimTag = m_graph->mapper.get<Dimension>(tag);
                rocRoller::Log::getLogger()->debug("  deallocate dimension: {}", dimTag);
                co_yield Instruction::Comment(concatenate("Deallocate ", dimTag));
                m_context->registerTagManager()->deleteTag(dimTag);
            }

            Generator<Instruction> operator()(int, Barrier const&, Transformer)
            {
                co_yield m_context->mem()->barrier();
            }

            Generator<Instruction> operator()(int tag, ComputeIndex const& ci, Transformer coords)
            {
                co_yield m_loadStoreTileGenerator.genComputeIndex(tag, ci, coords);
            }

            Generator<Instruction>
                operator()(int tag, SetCoordinate const& setCoordinate, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::SetCoordinate({}): {}",
                    tag,
                    Expression::toString(setCoordinate.value));

                auto connections = m_graph->mapper.getConnections(tag);
                AssertFatal(connections.size() == 1,
                            "Invalid SetCoordinate operation; coordinate missing.");
                co_yield Instruction::Comment(concatenate("SetCoordinate (",
                                                          tag,
                                                          "): Coordinate ",
                                                          connections[0].coordinate,
                                                          " = ",
                                                          setCoordinate.value));
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto init = m_graph->control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                co_yield generate(init, coords);

                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
            }

            Generator<Instruction> operator()(int tag, LoadLinear const& edge, Transformer coords)
            {
                Throw<FatalError>("LoadLinear present in kernel graph.");
            }

            Generator<Instruction> operator()(int tag, LoadTiled const& load, Transformer coords)
            {
                co_yield m_loadStoreTileGenerator.genLoadTile(tag, load, coords);
            }

            Generator<Instruction> operator()(int tag, LoadLDSTile const& load, Transformer coords)
            {
                co_yield m_loadStoreTileGenerator.genLoadLDSTile(tag, load, coords);
            }

            Generator<Instruction> operator()(int tag, LoadSGPR const& load, Transformer coords)
            {
                auto [userTag, user] = m_graph->getDimension<User>(tag);
                auto [vgprTag, vgpr] = m_graph->getDimension<VGPR>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadSGPR({}): User({}), VGPR({})",
                    tag,
                    userTag,
                    vgprTag);

                auto dst = m_context->registerTagManager()->getRegister(
                    vgprTag, Register::Type::Scalar, load.varType.dataType);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::Int64, 1);

                co_yield Instruction::Comment("GEN: LoadSGPR; user index");

                auto indexes = coords.reverse({userTag});
                co_yield generateOffset(
                    offset, indexes[0], dst->variableType().dataType, user.offset);

                Register::ValuePtr vPtr;

                {
                    Register::ValuePtr sPtr;
                    co_yield m_context->argLoader()->getValue(user.argumentName, sPtr);
                    co_yield m_context->copier()->ensureType(vPtr, sPtr, Register::Type::Scalar);
                }

                auto numBytes = DataTypeInfo::Get(dst->variableType()).elementSize;
                auto options  = BufferInstructionOptions();
                options.setGlc(load.glc);
                co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Scalar,
                                                dst,
                                                vPtr,
                                                offset,
                                                numBytes,
                                                "",
                                                false,
                                                nullptr,
                                                options);
            }

            Generator<Instruction> operator()(int tag, LoadVGPR const& load, Transformer coords)
            {
                auto [userTag, user] = m_graph->getDimension<User>(tag);
                auto [vgprTag, vgpr] = m_graph->getDimension<VGPR>(tag);

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR({}): User({}), VGPR({})",
                    tag,
                    userTag,
                    vgprTag);

                auto dst = m_context->registerTagManager()->getRegister(
                    vgprTag, Register::Type::Vector, load.varType.dataType);

                if(load.scalar)
                {
                    if(load.varType.isPointer())
                        co_yield loadVGPRFromScalarPointer(user, dst, coords);
                    else
                        co_yield loadVGPRFromScalarValue(user, dst, coords);
                }
                else
                {
                    co_yield loadVGPRFromGlobalArray(userTag, user, dst, coords);
                }
            }

            Generator<Instruction>
                loadVGPRFromScalarValue(User user, Register::ValuePtr vgpr, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar value");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar value");

                Register::ValuePtr s_value;
                co_yield m_context->argLoader()->getValue(user.argumentName, s_value);
                co_yield m_context->copier()->copy(vgpr, s_value, "Move value");
            }

            Generator<Instruction>
                loadVGPRFromScalarPointer(User user, Register::ValuePtr vgpr, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::CodeGenerator::LoadVGPR(): scalar pointer");
                co_yield Instruction::Comment("GEN: LoadVGPR; scalar pointer");

                Register::ValuePtr vPtr;

                {
                    Register::ValuePtr sPtr;
                    co_yield m_context->argLoader()->getValue(user.argumentName, sPtr);
                    co_yield m_context->copier()->ensureType(vPtr, sPtr, Register::Type::Vector);
                }

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, vPtr, nullptr, numBytes);
            }

            Generator<Instruction> loadVGPRFromGlobalArray(int                userTag,
                                                           User               user,
                                                           Register::ValuePtr vgpr,
                                                           Transformer        coords)
            {
                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                co_yield Instruction::Comment("GEN: LoadVGPR; user index");

                auto indexes = coords.reverse({userTag});
                co_yield generateOffset(
                    offset, indexes[0], vgpr->variableType().dataType, user.offset);

                Register::ValuePtr vPtr;

                {
                    Register::ValuePtr sPtr;
                    co_yield m_context->argLoader()->getValue(user.argumentName, sPtr);
                    co_yield m_context->copier()->ensureType(vPtr, sPtr, Register::Type::Vector);
                }

                auto numBytes = DataTypeInfo::Get(vgpr->variableType()).elementSize;
                co_yield m_context->mem()->load(
                    MemoryInstructions::MemoryKind::Flat, vgpr, vPtr, offset, numBytes);
            }

            Generator<Instruction> operator()(int tag, Multiply const& mult, Transformer coords)
            {
                auto [waveA_tag, waveA] = m_graph->getDimension<WaveTile>(
                    tag, Connections::typeArgument<WaveTile>(NaryArgument::LHS));
                auto [waveBTag, waveB] = m_graph->getDimension<WaveTile>(
                    tag, Connections::typeArgument<WaveTile>(NaryArgument::RHS));

                auto [macATag, macA] = m_graph->getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::LHS));
                auto [macBTag, macB] = m_graph->getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::RHS));

                AssertFatal(macA.sizes[1] == macB.sizes[0], "MacroTile size mismatch.");

                uint numElements = waveA.sizes[0] * waveB.sizes[1];
                uint wfs         = m_context->kernel()->wavefront_size();
                uint num_agpr    = numElements / wfs;

                auto [DTag, _D] = m_graph->getDimension<MacroTile>(
                    tag, Connections::typeArgument<MacroTile>(NaryArgument::DEST));

                auto D = m_context->registerTagManager()->getRegister(
                    DTag,
                    Register::Type::Accumulator,
                    DataType::Float,
                    num_agpr,
                    Register::AllocationOptions{.contiguousChunkWidth = 16});

                AssertFatal(D->allocation()->options().contiguousChunkWidth >= 16, "Should be 16");

                waveA.vgpr = m_context->registerTagManager()->getRegister(macATag);
                waveB.vgpr = m_context->registerTagManager()->getRegister(macBTag);

                auto A
                    = std::make_shared<Expression::Expression>(std::make_shared<WaveTile>(waveA));
                auto B
                    = std::make_shared<Expression::Expression>(std::make_shared<WaveTile>(waveB));

                auto matMul = std::make_shared<Expression::Expression>(
                    Expression::MatrixMultiply(A, B, D->expression()));

                co_yield Expression::generate(D, matMul, m_context);
            }

            Generator<Instruction> operator()(int tag, NOP const&, Transformer coords)
            {
                auto body = m_graph->control.getOutputNodeIndices<Body>(tag).to<std::set>();
                co_yield generate(body, coords);
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

            Generator<Instruction> operator()(int tag, StoreTiled const& store, Transformer coords)
            {
                co_yield m_loadStoreTileGenerator.genStoreTile(tag, store, coords);
            }

            Generator<Instruction>
                operator()(int tag, StoreLDSTile const& store, Transformer coords)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::CodeGenerator::StoreLDSTiled({})",
                                                   tag);
                co_yield Instruction::Comment("GEN: StoreLDSTile");

                co_yield m_loadStoreTileGenerator.genStoreLDSTile(tag, store, coords);
            }

            Generator<Instruction> operator()(int tag, StoreVGPR const& store, Transformer coords)
            {
                co_yield Instruction::Comment("GEN: StoreVGPR");

                auto [vgprTag, vgpr] = m_graph->getDimension<VGPR>(tag);
                auto [userTag, user] = m_graph->getDimension<User>(tag);

                auto src = m_context->registerTagManager()->getRegister(vgprTag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto indexes = coords.forward({userTag});

                co_yield Instruction::Comment("GEN: StoreVGPR; user index");
                co_yield generateOffset(
                    offset, indexes[0], src->variableType().dataType, user.offset);

                Register::ValuePtr vPtr;

                {
                    Register::ValuePtr sPtr;
                    co_yield m_context->argLoader()->getValue(user.argumentName, sPtr);
                    co_yield m_context->copier()->ensureType(vPtr, sPtr, Register::Type::Vector);
                }

                auto numBytes = DataTypeInfo::Get(src->variableType()).elementSize;
                co_yield m_context->mem()->store(
                    MemoryInstructions::MemoryKind::Flat, vPtr, src, offset, numBytes);
            }

            Generator<Instruction> operator()(int tag, StoreSGPR const& store, Transformer coords)
            {
                co_yield Instruction::Comment("GEN: StoreSGPR");

                auto [vgprTag, vgpr] = m_graph->getDimension<VGPR>(tag);
                auto [userTag, user] = m_graph->getDimension<User>(tag);

                auto src = m_context->registerTagManager()->getRegister(vgprTag);

                auto offset = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::Int64, 1);

                auto indexes = coords.forward({userTag});

                co_yield Instruction::Comment("GEN: StoreSGPR; user index");
                co_yield generateOffset(
                    offset, indexes[0], src->variableType().dataType, user.offset);

                Register::ValuePtr vPtr;

                {
                    Register::ValuePtr sPtr;
                    co_yield m_context->argLoader()->getValue(user.argumentName, sPtr);
                    co_yield m_context->copier()->ensureType(vPtr, sPtr, Register::Type::Scalar);
                }

                auto numBytes = DataTypeInfo::Get(src->variableType()).elementSize;
                auto options  = BufferInstructionOptions();
                options.setGlc(store.glc);
                co_yield m_context->mem()->store(MemoryInstructions::MemoryKind::Scalar,
                                                 vPtr,
                                                 src,
                                                 offset,
                                                 numBytes,
                                                 "",
                                                 false,
                                                 nullptr,
                                                 options);
            }

            Generator<Instruction> operator()(int, WaitZero const&, Transformer)
            {
                co_yield Instruction::Wait(WaitCount::Zero("Explicit WaitZero operation",
                                                           m_context->targetArchitecture()));
            }

        private:
            KernelGraphPtr m_graph;

            ContextPtr        m_context;
            AssemblyKernelPtr m_kernel;

            std::set<int> m_completedControlNodes;

            FastArithmetic         m_fastArith;
            LoadStoreTileGenerator m_loadStoreTileGenerator;
        };

        Generator<Instruction> generate(KernelGraph graph, AssemblyKernelPtr kernel)
        {
            TIMER(t, "KernelGraph::generate");
            auto graphPtr = std::make_shared<KernelGraph>(graph);

            if(Settings::getInstance()->get(Settings::LogGraphs))
                rocRoller::Log::getLogger()->debug("KernelGraph::generate(); DOT\n{}",
                                                   graphPtr->toDOT(true));

            auto visitor = CodeGeneratorVisitor(graphPtr, kernel);

            co_yield visitor.generate();
        }
    }
}
