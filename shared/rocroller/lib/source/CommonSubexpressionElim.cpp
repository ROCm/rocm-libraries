// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/CommonSubexpressionElim.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Expression
    {
        struct SubstituteValueVisitor
        {
            SubstituteValueVisitor(Register::ValuePtr target, Register::ValuePtr replacement)
                : m_target(target)
                , m_replacement(replacement)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) {
                    if(op)
                    {
                        op = call(op);
                    }
                });
                return std::make_shared<Expression>(std::move(cpy));
            }

            ExpressionPtr operator()(Register::ValuePtr const& value) const
            {
                return equivalent(value->expression(), m_target->expression())
                           ? m_replacement->expression()
                           : value->expression();
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};

                return std::visit(*this, *expr);
            }

        private:
            Register::ValuePtr m_target, m_replacement;
        };

        struct ReplaceValueWithExprVisitor
        {
            ReplaceValueWithExprVisitor(Register::ValuePtr target, ExpressionPtr replacement)
                : m_target(target)
                , m_replacement(replacement)
            {
            }

            template <CUnary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.arg)
                {
                    cpy.arg = call(expr.arg);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CBinary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.rhs)
                {
                    cpy.rhs = call(expr.rhs);
                }
                return std::make_shared<Expression>(cpy);
            }

            ExpressionPtr operator()(ScaledMatrixMultiply const& expr) const
            {
                ScaledMatrixMultiply cpy = expr;
                if(expr.matA)
                {
                    cpy.matA = call(expr.matA);
                }
                if(expr.matB)
                {
                    cpy.matB = call(expr.matB);
                }
                if(expr.matC)
                {
                    cpy.matC = call(expr.matC);
                }
                if(expr.scaleA)
                {
                    cpy.scaleA = call(expr.scaleA);
                }
                if(expr.scaleB)
                {
                    cpy.scaleB = call(expr.scaleB);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CTernary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                Expr cpy = expr;
                if(expr.lhs)
                {
                    cpy.lhs = call(expr.lhs);
                }
                if(expr.r1hs)
                {
                    cpy.r1hs = call(expr.r1hs);
                }
                if(expr.r2hs)
                {
                    cpy.r2hs = call(expr.r2hs);
                }
                return std::make_shared<Expression>(cpy);
            }

            template <CNary Expr>
            ExpressionPtr operator()(Expr const& expr) const
            {
                auto cpy = expr;
                std::ranges::for_each(cpy.operands, [this](auto& op) {
                    if(op)
                    {
                        op = call(op);
                    }
                });
                return std::make_shared<Expression>(std::move(cpy));
            }

            ExpressionPtr operator()(Register::ValuePtr const& value) const
            {
                return equivalent(value->expression(), m_target->expression())
                           ? m_replacement
                           : value->expression();
            }

            template <CValue Value>
            ExpressionPtr operator()(Value const& expr) const
            {
                return std::make_shared<Expression>(expr);
            }

            ExpressionPtr call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};

                return std::visit(*this, *expr);
            }

        private:
            Register::ValuePtr m_target;
            ExpressionPtr      m_replacement;
        };

        /**
         * @brief Replaces a register value with another in an expression
         *
         * @param expr Expression to find replacements for
         * @param target The target register to replace
         * @param replacement The new register to replace with
         * @return ExpressionPtr
         */
        ExpressionPtr substituteValue(ExpressionPtr      expr,
                                      Register::ValuePtr target,
                                      Register::ValuePtr replacement)
        {
            auto visitor = SubstituteValueVisitor(target, replacement);
            return visitor.call(expr);
        }

        struct ConsolidateSubExpressionVisitor
        {
            ConsolidateSubExpressionVisitor(ContextPtr context)
                : m_context(context)
            {
            }

            ExpressionTree operator()(Convert const& expr) const
            {
                auto tree = callUnary(expr);
                if(tree.empty())
                {
                    // Bail
                    return {};
                }

                AssertFatal(tree.back().deps.size() == 1);

                auto        depIdx = *tree.back().deps.begin();
                auto const& dep    = tree.at(depIdx);

                if(dep.reg->variableType() == expr.destinationType)
                {
                    // Simplify
                    tree.pop_back();
                    tree.back().reg = nullptr;
                    return tree;
                }
                else
                {
                    auto depTypeInfo = DataTypeInfo::Get(dep.reg->variableType());
                    auto expTypeInfo = DataTypeInfo::Get(expr.destinationType);

                    if(depTypeInfo.isIntegral && expTypeInfo.isIntegral
                       && depTypeInfo.elementBits == expTypeInfo.elementBits)
                    {
                        auto valueRange = iota(0ul, dep.reg->valueCount()).to<std::vector>();

                        auto reg = dep.reg->element(valueRange);
                        reg->setVariableType(expr.destinationType);

                        auto regWithName = tree.back().reg ? tree.back().reg : reg;
                        reg->setName(regWithName->name() + " convertInPlace");

                        Log::trace(
                            "Forwarding {} to {}", dep.reg->description(), reg->description());
                        Log::trace("Forwarding {} to {}",
                                   (void*)dep.reg->allocation().get(),
                                   (void*)reg->allocation().get());

                        tree.back().reg = reg;
                    }
                    else if(depTypeInfo.isIntegral && expTypeInfo.isIntegral
                            && depTypeInfo.elementBits == 64 && expTypeInfo.elementBits == 32)
                    {
                        std::vector<int> indices;
                        for(int i = 0; i < depTypeInfo.registerCount;
                            i += depTypeInfo.elementBits / Register::bitsPerRegister)
                        {
                            indices.push_back(i);
                        }
                        auto reg = dep.reg->subset(indices);
                        reg->setVariableType(expr.destinationType);
                        auto regWithName = tree.back().reg ? tree.back().reg : reg;
                        reg->setName(regWithName->name() + " convertForwardSubset");
                        tree.back().reg = reg;
                    }
                }

                return tree;
            }

            ExpressionTree operator()(Reinterpret const& expr) const
            {
                auto tree = callUnary(expr);
                if(tree.empty())
                {
                    // Bail
                    return {};
                }

                AssertFatal(tree.back().deps.size() == 1);
                auto        depIdx = *tree.back().deps.begin();
                auto const& dep    = tree.at(depIdx);

                auto depTypeInfo = DataTypeInfo::Get(dep.reg->variableType());
                auto expTypeInfo = DataTypeInfo::Get(expr.destinationType);

                AssertFatal(depTypeInfo.elementBytes == expTypeInfo.elementBytes,
                            "Reinterpret requires same size types: source type ",
                            toString(dep.reg->variableType().dataType),
                            " (",
                            depTypeInfo.elementBytes,
                            " bytes) != destination type ",
                            toString(expr.destinationType),
                            " (",
                            expTypeInfo.elementBytes,
                            " bytes)");

                auto registerCount = dep.reg->registerCount();
                AssertFatal(registerCount == expTypeInfo.registerCount,
                            "Reinterpret requires same number of registers: source type ",
                            toString(dep.reg->variableType().dataType),
                            " (",
                            registerCount,
                            " registers) != destination type ",
                            toString(expr.destinationType),
                            " (",
                            expTypeInfo.registerCount,
                            " registers)");

                // Sets the destination register for Reinterpret as the argument register
                // but with the reinterpreted type
                Register::ValuePtr reg
                    = std::make_shared<Register::Value>(dep.reg->allocation(),
                                                        dep.reg->regType(),
                                                        expr.destinationType,
                                                        dep.reg->allocationCoord());
                tree.back().reg = reg;

                dep.reg->setName(dep.reg->name() + " reinterpret");
                Log::trace("Reinterpreting {} to {}",
                           dep.reg->description(),
                           tree.back().reg->description());

                return tree;
            }

            ExpressionTree operator()(BitFieldExtract const& expr) const
            {
                auto tree = callUnary(expr);
                if(tree.empty())
                    return {};

                AssertFatal(tree.back().deps.size() == 1, ShowValue(tree.back().deps.size()));

                auto        deps               = tree.back().deps;
                auto        consolidationCount = tree.back().consolidationCount;
                auto        depIdx             = *deps.begin();
                auto const& dep                = tree.at(depIdx);

                // Try to simplify BitFieldExtract to a subset
                if(auto subset = bfeToSubset(expr, dep.reg))
                {
                    Log::trace("Eliminating BitFieldExtract of full registers {}",
                               subset.value()->description());
                    tree.pop_back();
                    auto value = subset.value();
                    tree.push_back({value, value->expression(), deps, consolidationCount});
                }

                return tree;
            }

            template <CUnary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                return callUnary(expr);
            }

            template <CUnary Expr>
            ExpressionTree callUnary(Expr const& expr) const
            {
                ExpressionTree tree;
                std::set<int>  deps;
                int            loc;

                tree = generateTree(expr.arg);
                if(tree.empty())
                    return {};

                loc = tree.size() - 1;
                deps.insert(loc);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.back()) ? tree.back().reg->expression() : tree.back().expr,
                        getComment(expr),
                    }
                                                     .copyParams(expr)),
                    deps,
                    tree.back().consolidationCount,
                });

                return tree;
            }

            template <CBinary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                ExpressionTree rhsTree, tree;
                std::set<int>  deps;
                int            lhsLoc, rhsLoc;
                int            consolidationCount = 0;

                tree = generateTree(expr.lhs);
                if(tree.empty())
                    return {};

                lhsLoc = tree.size() - 1;
                deps.insert(lhsLoc);
                int lhsSize = tree.size();

                rhsTree = generateTree(expr.rhs);
                if(rhsTree.empty())
                    return {};

                auto combo = combine(tree, rhsTree);
                tree       = std::get<0>(combo);
                rhsLoc     = std::get<1>(combo);

                deps.insert(rhsLoc);

                consolidationCount += tree.at(lhsLoc).consolidationCount;
                consolidationCount += tree.at(rhsLoc).consolidationCount;
                consolidationCount += std::get<2>(combo);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.at(lhsLoc)) ? tree.at(lhsLoc).reg->expression()
                                                     : tree.at(lhsLoc).expr,
                        canUseAsReg(tree.at(rhsLoc)) ? tree.at(rhsLoc).reg->expression()
                                                     : tree.at(rhsLoc).expr,
                        getComment(expr),
                    }
                                                     .copyParams(expr)),
                    deps,
                    consolidationCount,
                });

                return tree;
            }

            template <CTernary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                ExpressionTree r1hsTree, r2hsTree, tree;
                std::set<int>  deps;
                int            lhsLoc, r1hsLoc, r2hsLoc;
                int            consolidationCount = 0;

                tree = generateTree(expr.lhs);
                if(tree.empty())
                    return {};

                lhsLoc = tree.size() - 1;
                deps.insert(lhsLoc);
                int lhsSize = tree.size();

                r1hsTree = generateTree(expr.r1hs);
                if(r1hsTree.empty())
                    return {};

                r2hsTree = generateTree(expr.r2hs);
                if(r2hsTree.empty())
                    return {};

                auto combo1 = combine(tree, r1hsTree);
                tree        = std::get<0>(combo1);
                r1hsLoc     = std::get<1>(combo1);

                deps.insert(r1hsLoc);

                auto combo2 = combine(tree, r2hsTree);
                tree        = std::get<0>(combo2);
                r2hsLoc     = std::get<1>(combo2);

                deps.insert(r2hsLoc);

                consolidationCount += tree.at(lhsLoc).consolidationCount;
                consolidationCount += tree.at(r1hsLoc).consolidationCount;
                consolidationCount += tree.at(r2hsLoc).consolidationCount;
                consolidationCount += std::get<2>(combo1);
                consolidationCount += std::get<2>(combo2);

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{
                        canUseAsReg(tree.at(lhsLoc)) ? tree.at(lhsLoc).reg->expression()
                                                     : tree.at(lhsLoc).expr,
                        canUseAsReg(tree.at(r1hsLoc)) ? tree.at(r1hsLoc).reg->expression()
                                                      : tree.at(r1hsLoc).expr,
                        canUseAsReg(tree.at(r2hsLoc)) ? tree.at(r2hsLoc).reg->expression()
                                                      : tree.at(r2hsLoc).expr,
                    }
                                                     .copyParams(expr)),
                    deps,
                    consolidationCount,
                });

                setComment(tree.back().expr, getComment(expr));

                return tree;
            }

            ExpressionTree operator()(BitfieldCombine const& expr) const
            {
                AssertFatal(
                    false,
                    "BitfieldCombine should be lowered to shift and bitwise AND/OR epxressions");
                return {};
            }

            template <CNary Expr>
            ExpressionTree operator()(Expr const& expr) const
            {
                std::vector<ExpressionTree> operandTrees;
                std::set<int>               deps;
                std::vector<int>            operandLocs;
                int                         consolidationCount = 0;

                if(expr.operands.empty())
                {
                    return {};
                }

                operandTrees.reserve(expr.operands.size());
                operandLocs.reserve(expr.operands.size());

                for(auto const& operand : expr.operands)
                {
                    ExpressionTree operandTree = generateTree(operand);
                    if(operandTree.empty())
                    {
                        return {};
                    }
                    operandTrees.push_back(std::move(operandTree));
                }

                ExpressionTree tree = operandTrees.at(0);
                operandLocs.push_back(tree.size() - 1);
                deps.insert(tree.size() - 1);

                for(size_t i = 1; i < expr.operands.size(); ++i)
                {
                    auto combo = combine(tree, operandTrees.at(i));
                    tree       = std::get<0>(combo);
                    int loc    = std::get<1>(combo);
                    operandLocs.push_back(loc);
                    deps.insert(loc);
                    consolidationCount += std::get<2>(combo);
                }

                for(size_t i = 0; i < expr.operands.size(); ++i)
                {
                    int loc = operandLocs.at(i);
                    consolidationCount += tree.at(loc).consolidationCount;
                }

                std::vector<ExpressionPtr> nodeOperands;
                nodeOperands.reserve(expr.operands.size());
                for(size_t i = 0; i < expr.operands.size(); ++i)
                {
                    int loc = operandLocs.at(i);
                    nodeOperands.push_back(canUseAsReg(tree.at(loc))
                                               ? tree.at(loc).reg->expression()
                                               : tree.at(loc).expr);
                }

                tree.push_back(ExpressionNode{
                    nullptr,
                    std::make_shared<Expression>(Expr{std::move(nodeOperands)}.copyParams(expr)),
                    deps,
                    consolidationCount,
                });

                setComment(tree.back().expr, getComment(expr));

                return tree;
            }

            ExpressionTree operator()(ScaledMatrixMultiply const& expr) const
            {
                return {};
            }

            ExpressionTree operator()(WaveTilePtr const& value) const
            {
                return {};
            }

            ExpressionTree operator()(DataFlowTag const& value) const
            {
                return {};
            }

            ExpressionTree operator()(PositionalArgument const& value) const
            {
                return {};
            }

            ExpressionTree operator()(Register::ValuePtr const& value) const
            {
                return {{value, value->expression(), {}, 0}};
            }

            template <CValue Value>
            ExpressionTree operator()(Value const& expr) const
            {
                return {{nullptr, std::make_shared<Expression>(expr), {}, 0}};
            }

            ExpressionTree call(ExpressionPtr expr) const
            {
                if(!expr)
                    return {};
                return std::visit(*this, *expr);
            }

            ExpressionTree call(Expression const& expr) const
            {
                return std::visit(*this, expr);
            }

        private:
            ContextPtr m_context;

            /**
             * @brief Attempts to simplify a BitFieldExtract to a register subset
             *
             * @param expr The BitFieldExtract expression
             * @param reg The source register
             * @return std::optional<Register::ValuePtr> The subset register if simplification is possible, nullopt otherwise
             */
            std::optional<Register::ValuePtr> bfeToSubset(BitFieldExtract const& expr,
                                                          Register::ValuePtr     reg) const
            {
                // Check if extraction is aligned to register boundaries
                if(expr.offset % Register::bitsPerRegister != 0
                   || expr.width % Register::bitsPerRegister != 0)
                    return std::nullopt;

                uint registerOffset = expr.offset / Register::bitsPerRegister;
                uint registerCount  = expr.width / Register::bitsPerRegister;

                // Only simplify if the register count matches the expected output data type
                if(DataTypeInfo::Get(expr.outputDataType).registerCount != registerCount)
                    return std::nullopt;

                AssertFatal(registerOffset + registerCount <= reg->registerCount(),
                            "BitFieldExtract offset and width are out of bounds: ",
                            ShowValue(registerOffset),
                            ShowValue(registerCount),
                            ShowValue(reg->registerCount()));

                // Create subset of registers
                std::vector<int> indices(registerCount);
                std::iota(indices.begin(), indices.end(), registerOffset);
                auto subset = reg->subset(indices);
                subset->setVariableType(expr.outputDataType);

                return subset;
            }

            Register::ValuePtr resultPlaceholder(ResultType const& resType,
                                                 bool              allowSpecial = true) const
            {
                // Obtain the register count by dividing the value count of the result type by its packing
                size_t count = resType.valueCount
                               / (resType.varType.dataType == DataType::None
                                      ? 1
                                      : DataTypeInfo::Get(resType.varType).packing);

                if(Register::IsSpecial(resType.regType) && resType.varType == DataType::Bool)
                {
                    if(allowSpecial)
                        return m_context->getSCC();
                    else
                        return Register::Value::Placeholder(
                            m_context, Register::Type::Scalar, resType.varType, count);
                }
                return Register::Value::Placeholder(
                    m_context, resType.regType, resType.varType, count);
            }

            /**
             * @brief Generate a tree from an expression's argument. Creates placeholder.
             *
             * @param expr Expression to decompose
             * @return ExpressionTree
             */
            ExpressionTree generateTree(ExpressionPtr expr) const
            {
                ExpressionTree tree = {};

                if(expr)
                {
                    tree = call(expr);
                    if(tree.empty())
                    {
                        // Bail
                        return {};
                    }
                    if(tree.back().reg == nullptr)
                    {
                        tree.back().reg = resultPlaceholder(resultType(expr), sccAvailable(tree));
                    }
                }

                return tree;
            }

            /**
             * @brief Determine whether we can use the register of the node, or if we should use the expression instead
             *
             * @param target Node to analyze
             */
            bool canUseAsReg(ExpressionNode const& target) const
            {
                return !(target.reg && target.reg->regType() == Register::Type::Literal)
                       && !(target.expr
                            && (std::holds_alternative<CommandArgumentPtr>(*target.expr)
                                || std::holds_alternative<AssemblyKernelArgumentPtr>(*target.expr)
                                || std::holds_alternative<WaveTilePtr>(*target.expr)));
            }

            /**
             * @brief Determine if we should note this node in consolidation count tracking
             *
             * @param target Node to analyze
             */
            bool shouldTrack(ExpressionNode const& target) const
            {
                return !(target.reg && target.reg->regType() == Register::Type::Literal)
                       && !(target.expr
                            && (std::holds_alternative<Register::ValuePtr>(*target.expr)
                                || std::holds_alternative<CommandArgumentPtr>(*target.expr)
                                || std::holds_alternative<AssemblyKernelArgumentPtr>(*target.expr)
                                || std::holds_alternative<WaveTilePtr>(*target.expr)));
            }

            /**
             * @brief Check if the SCC register is available to use
             *
             * @param tree The expression tree to check
             * @return False if SCC is not to be used
             */
            bool sccAvailable(ExpressionTree const& tree) const
            {
                return false;
            }

            /**
             * @brief Combine two ExpressionTrees
             *
             * @param tree1
             * @param tree2
             * @return std::tuple<ExpressionTree, int> Combined tree, the index of tree2's root in resulting tree
             */
            std::tuple<ExpressionTree, int, int> combine(ExpressionTree const& tree1,
                                                         ExpressionTree const& tree2) const
            {
                auto tree    = tree1; // Tree to graft onto
                auto rhsTree = tree2; // Tree to merge in
                int  lhsSize = tree.size();

                // Replace the RHS dependencies with what they would be if the trees were combined without elimination
                for(auto& rhsNode : rhsTree)
                {
                    std::set<int> newDeps;
                    for(auto dep : rhsNode.deps)
                    {
                        newDeps.insert(dep + lhsSize);
                    }
                    rhsNode.deps = newDeps;
                }

                // The location of the root of the right tree after combining
                int rhsLoc = -1; // Initialize with invalid value

                // Track how many nodes were skipped
                int tracker = 0;

                auto sccExpr = m_context->getSCC()->expression();
                auto lhsHasSCC
                    = std::find_if(tree.begin(),
                                   tree.end(),
                                   [sccExpr](auto origNode) {
                                       return equivalent(origNode.reg->expression(), sccExpr);
                                   })
                      != tree.end();

                // Loop through the RHS
                for(auto rhsNodeIt = rhsTree.begin(); rhsNodeIt != rhsTree.end(); rhsNodeIt++)
                {
                    // Find a node that has an equivalent expression in the growing tree
                    auto it = std::find_if(tree.begin(), tree.end(), [&rhsNodeIt](auto origNode) {
                        return equivalent(rhsNodeIt->expr, origNode.expr);
                    });

                    bool rhsIsSCC = equivalent(rhsNodeIt->reg->expression(),
                                               m_context->getSCC()->expression());

                    if(it != tree.end())
                    {
                        // If we found a valid expression to eliminate
                        auto rhsIndex = rhsNodeIt - rhsTree.begin();
                        auto newIndex = it - tree.begin();

                        if(rhsTree.size() - 1 == rhsIndex)
                        {
                            // If we're on the last RHS node, the root is no longer the last element
                            rhsLoc = newIndex;
                        }

                        // Loop through remaining RHS nodes
                        for(auto scanIt = rhsNodeIt + 1; scanIt != rhsTree.end(); scanIt++)
                        {
                            // Find relevant dependencies
                            if(scanIt->deps.contains(rhsIndex + lhsSize))
                            {
                                // Replace expression
                                scanIt->expr
                                    = substituteValue(scanIt->expr, rhsNodeIt->reg, it->reg);
                                // Replace dependency
                                scanIt->deps.erase(rhsIndex + lhsSize);
                                scanIt->deps.insert(newIndex);
                            }
                        }
                        if(shouldTrack(*rhsNodeIt))
                        {
                            tracker++;
                        }

                        rhsNodeIt->reg = nullptr;
                        continue;
                    }

                    if(rhsIsSCC && lhsHasSCC)
                    {
                        // Found 2 SCCs! Fixup
                        rhsNodeIt->reg = resultPlaceholder(resultType(rhsNodeIt->expr), false);
                    }

                    // Node is good to add to tree
                    tree.push_back(*rhsNodeIt);

                    auto rhsIndex = rhsNodeIt - rhsTree.begin();

                    // Update following RHS nodes if necessary
                    for(auto scanIt = rhsNodeIt + 1; scanIt != rhsTree.end(); scanIt++)
                    {
                        if(rhsIndex + lhsSize != tree.size() - 1
                           && scanIt->deps.contains(rhsIndex + lhsSize))
                        {
                            scanIt->deps.erase(rhsIndex + lhsSize);
                            scanIt->deps.insert(tree.size() - 1);
                        }
                    }
                }

                // The last element is the RHS root, unless it was otherwise updated
                if(rhsLoc == -1)
                    rhsLoc = tree.size() - 1;

                return {tree, rhsLoc, tracker};
            }
        };

        ExpressionTree consolidateSubExpressions(ExpressionPtr expr, ContextPtr context)
        {
            if(!expr)
                return {};
            auto visitor = ConsolidateSubExpressionVisitor(context);
            auto rv      = visitor.call(*expr);

            updateDistances(rv);
            updatePriorityOrder(rv);

            return rv;
        }

        int getConsolidationCount(ExpressionTree const& tree)
        {
            return tree.back().consolidationCount;
        }

        ExpressionPtr rebuildExpressionHelper(ExpressionTree const& tree, int index)
        {
            auto root = tree.at(index);
            if(root.deps.empty())
            {
                return root.expr;
            }

            ExpressionPtr modifiedExpr = root.expr;
            for(auto dep : root.deps)
            {
                auto                        replacementExpr = rebuildExpressionHelper(tree, dep);
                ReplaceValueWithExprVisitor v(tree.at(dep).reg, replacementExpr);
                modifiedExpr = v.call(modifiedExpr);
            }
            return modifiedExpr;
        }

        ExpressionPtr rebuildExpression(ExpressionTree const& tree)
        {
            if(tree.empty())
            {
                return nullptr;
            }
            return rebuildExpressionHelper(tree, tree.size() - 1);
        }

        void updateDistances(ExpressionTree& tree)
        {
            if(tree.size() < 2)
                return;

            for(int idx = 0; idx < tree.size(); idx++)
            {
                auto& node = tree[idx];
                AssertFatal(
                    node.deps.empty() || *node.deps.rbegin() < idx,
                    "tree is no longer in topological order! This function needs to be updated!");
                node.distanceFromRoot = 0;
            }

            for(auto const& nodeA : std::ranges::views::reverse(tree))
            {
                int newLevel = nodeA.distanceFromRoot + 1;

                for(int idxB : nodeA.deps)
                {
                    if(tree.at(idxB).distanceFromRoot < newLevel)
                        tree[idxB].distanceFromRoot = newLevel;
                }
            }
        }

        namespace
        {
            /**
             * @brief Computes Sethi-Ullman inspired weights for each node.
             *
             * Weights approximate the minimum number of registers needed to evaluate
             * a subtree. The algorithm processes nodes bottom-up:
             *
             * - Leaf nodes: weight = 0 for literals (immediate operands),
             *               weight = 1 for values needing a register
             * - Internal nodes: sort dependency weights descending, then
             *               weight = max(w[0], w[1]+1, w[2]+2, w[3]+3, ...)
             */
            std::vector<int> ComputeWeights(ExpressionTree const& tree)
            {
                std::vector<int> weights(tree.size());

                // Process in topological order (the tree is already sorted this way)
                for(size_t i = 0; i < tree.size(); ++i)
                {
                    auto const& node = tree.at(i);

                    // Leaf node - base case
                    if(node.deps.empty())
                    {
                        weights[i] = (node.regType() == Register::Type::Literal) ? 0 : 1;
                        continue;
                    }

                    // Collect and sort dependency weights (heaviest first)
                    std::vector<int> depWeights;
                    depWeights.reserve(node.deps.size());
                    for(int dep : node.deps)
                        depWeights.push_back(weights[dep]);
                    std::sort(depWeights.begin(), depWeights.end(), std::greater<int>());

                    // Child j needs +j registers to hold results of children 0..j-1.
                    weights[i] = depWeights[0];
                    for(size_t j = 1; j < depWeights.size(); ++j)
                        weights[i] = std::max(weights[i], depWeights[j] + static_cast<int>(j));
                }

                return weights;
            }

            /**
             * @brief Recursive helper for computing Sethi-Ullman inspired traversal order.
             *
             * Visits nodes in post-order, with heavier children visited first.
             */
            void ComputeOrderRecursive(ExpressionTree const&   tree,
                                       std::vector<int> const& weights,
                                       std::vector<int>&       order,
                                       int                     nodeIdx,
                                       int&                    orderCounter)
            {
                if(order[nodeIdx] != -1) // already visited
                    return;

                auto const& node = tree.at(nodeIdx);

                // Get dependencies sorted by weight (heaviest first)
                std::vector<int> deps(node.deps.begin(), node.deps.end());
                std::sort(deps.begin(), deps.end(), [&weights](int a, int b) {
                    if(weights[a] != weights[b])
                        return weights[a] > weights[b]; // heaviest first
                    return a < b; // tie-breaker
                });

                for(int dep : deps)
                    ComputeOrderRecursive(tree, weights, order, dep, orderCounter);

                // Assign order to this node after all children are processed
                order[nodeIdx] = orderCounter++;
            }
        } // namespace

        void updatePriorityOrder(ExpressionTree& tree)
        {
            if(tree.empty())
                return;

            // Verify topological order
            for(size_t idx = 0; idx < tree.size(); idx++)
            {
                auto const& node = tree[idx];
                if(node.deps.empty())
                    continue;

                int maxDep = 0;
                for(int dep : node.deps)
                    maxDep = std::max(maxDep, dep);

                AssertFatal(maxDep < static_cast<int>(idx),
                            "ExpressionTree is not in topological order!!",
                            ShowValue(maxDep),
                            ShowValue(idx));
            }

            auto weights = ComputeWeights(tree);

            // Compute traversal order: visit heavier subtrees first
            std::vector<int> order(tree.size(), -1); // -1 means not visited
            int              orderCounter = 0;

            // Start from the root (last node in the tree)
            int rootIdx = static_cast<int>(tree.size()) - 1;
            ComputeOrderRecursive(tree, weights, order, rootIdx, orderCounter);

            // Store the computed order in each node
            for(size_t i = 0; i < tree.size(); ++i)
            {
                AssertFatal(order[i] != -1,
                            "Node not visited in ExpressionTree!!",
                            ShowValue(i),
                            ShowValue(order[i]));
                tree[i].priorityOrder = order[i];
            }
        }

        std::string statistics(ExpressionTree tree)
        {
            std::string rv = fmt::format("Expression tree with {} nodes.", tree.size());
            updateDistances(tree);

            std::map<int, int> levelCounts;
            std::map<int, int> levelDeltas;
            for(int idx = 0; idx < tree.size(); idx++)
            {
                auto d = tree[idx].distanceFromRoot;
                levelCounts[d]++;

                for(auto idx2 : tree[idx].deps)
                {
                    auto d2 = tree.at(idx2).distanceFromRoot;
                    levelDeltas[d2 - d]++;
                }
            }

            rv += "Level Value Counts:\n";
            for(auto [level, count] : std::ranges::views::reverse(levelCounts))
                rv += fmt::format("{}: {}\n", level, count);

            rv += "Level Delta Counts:\n";
            for(auto [level, count] : std::ranges::views::reverse(levelDeltas))
                rv += fmt::format("{}: {}\n", level, count);

            return rv;
        }

        std::string describe(ExpressionNode const& node)
        {
            std::string desc = "nullptr";
            if(node.reg)
                desc = node.reg->description();
            return fmt::format(
                "{} = {} (level {})", desc, toString(node.expr), node.distanceFromRoot);
        }

        Register::Type ExpressionNode::regType() const
        {
            if(reg != nullptr)
                return reg->regType();
            if(expr != nullptr)
                return resultRegisterType(expr);
            return Register::Type::Count;
        }

        std::string toDOT(ExpressionTree const& tree)
        {
            std::ostringstream msg;
            msg << "digraph {" << std::endl;
            for(size_t i = 0; i < tree.size(); i++)
            {
                msg << fmt::format("\"node{}\"[label=\"{}\"]\n", i, describe(tree[i]));
            }

            for(size_t i = 0; i < tree.size(); i++)
            {
                for(auto j : tree[i].deps)
                    msg << fmt::format("\"node{}\" -> \"node{}\"\n", i, j);
            }

            msg << "}\n";

            return msg.str();
        }
    }
}
