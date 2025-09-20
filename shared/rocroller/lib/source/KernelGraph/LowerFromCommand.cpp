/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*
 * Command to KernelGraph translator
 */

#include <variant>
#include <vector>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/BlockScale.hpp>
#include <rocRoller/Operations/Command.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace CoordinateGraph;
        using namespace ControlGraph;

        /**
         * @brief Promote inputs to an appropriate output.
         *
         * For example, given VGPR and Linear inputs, output should be Linear.
         */
        Dimension
            promoteDimensions(Operations::OperationTag                                        cTag,
                              rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph const& graph,
                              std::vector<int> const&                                         dims)
        {
            Dimension rv = VGPR(cTag);
            for(auto tag : dims)
            {
                auto element = graph.getElement(tag);

                std::visit(
                    rocRoller::overloaded{
                        [](VGPR const& op) {},
                        [&](Linear const& op) {
                            AssertFatal(
                                !std::holds_alternative<MacroTile>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = Linear(cTag);
                        },
                        [&](MacroTile const& op) {
                            AssertFatal(
                                !std::holds_alternative<Linear>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = MacroTile(cTag);
                        },
                        [](auto const& op) {
                            Throw<FatalError>("Invalid argument of element operation.");
                        }},
                    std::get<Dimension>(element));
            }
            return rv;
        }

        /**
         * @brief Command to KernelGraph translator.
         */
        struct TranslateVisitor
        {
            TranslateVisitor()
            {
                m_kernel = m_graph.control.addElement(Kernel());
            }

            /**
             * @brief Translate `T_Load_Linear` to `LoadLinear`.
             *
             * Coordinates for `T_Load_Linear` are:
             *
             *           Split                         Flatten
             *     User ------> { SubDimension, ... } --------> Linear
             *
             * and:
             *
             *           DataFlow
             *     User ---------> Linear.
             *
             * For example:
             *
             *     T_Load_Linear[dataflow=1, sizes={16, 32}, strides={32, 1}]
             *
             * becomes:
             *
             *           Split
             *     User ------> { SubDimension(0, size=16, stride=32),
             *                    SubDimension(1, size=32, stride=1) }
             *
             *           Flatten
             *          --------> Linear
             *
             * The reverse transform of a `Split` edge honours the
             * users strides.
             *
             * The resulting `Linear` dimension is contiguous (because
             * of the `Flatten` edge).
             */
            void operator()(Operations::T_Load_Linear const& tload)
            {
                auto tensor  = m_command->getOperation<Operations::Tensor>(tload.getSrcTag());
                auto sizes   = tensor.sizes();
                auto strides = tensor.strides();

                auto totalSizeExpr = std::make_shared<Expression::Expression>(sizes[0]);

                auto user = m_graph.coordinates.addElement(
                    User(tload.getTag(),
                         tensor.data()->name(),
                         std::make_shared<Expression::Expression>(tensor.limit())));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto sizeExpr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto strideExpr = std::make_shared<Expression::Expression>(strides[i]);

                    dims.push_back(
                        m_graph.coordinates.addElement(SubDimension(i, sizeExpr, strideExpr)));
                    if(i > 0)
                        totalSizeExpr = totalSizeExpr * sizeExpr;
                }

                m_graph.coordinates.addElement(Split(), std::vector<int>{user}, dims);

                auto unit_stride = Expression::literal(1u);
                auto linear      = m_graph.coordinates.addElement(
                    Linear(tload.getTag(), totalSizeExpr, unit_stride));

                m_graph.coordinates.addElement(Flatten(), dims, std::vector<int>{linear});
                m_graph.coordinates.addElement(DataFlow(), {user}, {linear});

                auto load = m_graph.control.addElement(LoadLinear(tensor.dataType()));
                m_graph.control.addElement(Body(), {m_kernel}, {load});

                m_graph.mapper.connect<User>(load, user);
                m_graph.mapper.connect<Linear>(load, linear);

                m_op.insert_or_assign(tload.getTag(), load);
                m_dim.insert_or_assign(tload.getTag(), linear);
            }

            /**
             * @brief Translate `T_Load_Scalar` to  `LoadVGPR`.
             *
             * Coordinates for `T_Load_Scalar` are:
             *
             *           DataFlow
             *     User ---------> VGPR
             *
             */
            void operator()(Operations::T_Load_Scalar const& tload)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Load_Scalar");

                auto scalar = m_command->getOperation<Operations::Scalar>(tload.getSrcTag());
                auto user
                    = m_graph.coordinates.addElement(User(tload.getTag(), scalar.data()->name()));
                auto vgpr = m_graph.coordinates.addElement(VGPR(tload.getTag()));

                m_graph.coordinates.addElement(DataFlow(), {user}, {vgpr});

                auto load = m_graph.control.addElement(LoadVGPR(scalar.variableType(), true));
                m_graph.control.addElement(Body(), {m_kernel}, {load});

                m_graph.mapper.connect<User>(load, user);
                m_graph.mapper.connect<VGPR>(load, vgpr);

                m_op.insert_or_assign(tload.getTag(), load);
                m_dim.insert_or_assign(tload.getTag(), vgpr);
            }

            /**
             * @brief Translate `T_Load_Tiled` to `LoadTiled`.
             *
             * Coordinates for `T_Load_Tiled` are:
             *
             *           Split                         ConstructMacroTile
             *     User ------> { SubDimension, ... } -------------------> MacroTile
             *
             * and:
             *
             *           DataFlow
             *     User ---------> MacroTile.
             *
             * For example:
             *
             *     T_Load_Tiled[dataflow=1, sizes={16, 32}, strides={32, 1}]
             *
             * becomes:
             *
             *           Split
             *     User ------> { SubDimension(0 size=16, stride=32),
             *                    SubDimension(1, size=32, stride=1) }
             *
             *           ConstructMacroTile
             *          -------------------> MacroTile.
             *
             * The sizes, layout etc of the tile do not need to be
             * known at this stage.
             */
            void operator()(Operations::T_Load_Tiled const& tload)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Load_Tiled");

                auto tensor = m_command->getOperation<Operations::Tensor>(tload.getSrcTag());

                auto const sizes          = tensor.sizes();
                auto const strides        = tensor.strides();
                auto const literalStrides = tensor.literalStrides();

                auto user = m_graph.coordinates.addElement(
                    User(tload.getTag(),
                         tensor.data()->name(),
                         std::make_shared<Expression::Expression>(tensor.limit())));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto sizeExpr = std::make_shared<Expression::Expression>(sizes[i]);
                    std::shared_ptr<Expression::Expression> strideExpr;
                    if(literalStrides.size() > i && literalStrides[i] > 0)
                    {
                        strideExpr = std::make_shared<Expression::Expression>(literalStrides[i]);
                    }
                    else
                    {
                        strideExpr = std::make_shared<Expression::Expression>(strides[i]);
                    }

                    auto dim
                        = m_graph.coordinates.addElement(SubDimension(i, sizeExpr, strideExpr));
                    dims.push_back(dim);
                }

                auto tiled
                    = m_graph.coordinates.addElement(MacroTile(tload.getTag(), sizes.size()));

                m_graph.coordinates.addElement(Split(), std::vector<int>{user}, dims);
                m_graph.coordinates.addElement(ConstructMacroTile(), dims, std::vector<int>{tiled});
                m_graph.coordinates.addElement(DataFlow(), {user}, {tiled});

                auto load = m_graph.control.addElement(LoadTiled(tensor.variableType()));
                m_graph.control.addElement(Body(), {m_kernel}, {load});

                m_graph.mapper.connect<User>(load, user);
                m_graph.mapper.connect<MacroTile>(load, tiled);

                m_op.insert_or_assign(tload.getTag(), load);
                m_dim.insert_or_assign(tload.getTag(), tiled);
            }

            /**
             * @brief Translate `T_Store_Linear` to `StoreLinear`.
             *
             * Coordinates for `T_Store_Linear` are:
             *
             *             Split                         Join
             *     Linear ------> { SubDimension, ... } -----> User
             *
             * and:
             *
             *             DataFlow
             *     Linear ---------> User.
             *
             * The `Join` edge honours user strides.
             */
            void operator()(Operations::T_Store_Linear const& tstore)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Store_Linear");
                AssertFatal(m_op.count(tstore.getSrcTag()) > 0,
                            "Unknown command tag",
                            ShowValue(tstore.getSrcTag()));

                auto tensor = m_command->getOperation<Operations::Tensor>(tstore.getDstTag());

                std::vector<int> dims;
                auto             strides = tensor.strides();
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    auto strideExpr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim = m_graph.coordinates.addElement(SubDimension(i, nullptr, strideExpr));
                    dims.push_back(dim);
                }

                auto linear = m_dim.at(tstore.getSrcTag());
                auto user   = m_graph.coordinates.addElement(
                    User(tstore.getSrcTag(),
                         tensor.data()->name(),
                         std::make_shared<Expression::Expression>(tensor.limit())));

                m_graph.coordinates.addElement(Split(), std::vector<int>{linear}, dims);
                m_graph.coordinates.addElement(Join(), dims, std::vector<int>{user});
                m_graph.coordinates.addElement(DataFlow(), {linear}, {user});

                auto store = m_graph.control.addElement(StoreLinear());
                auto last  = m_op.at(tstore.getSrcTag());
                m_graph.control.addElement(Sequence(), {last}, {store});

                m_graph.mapper.connect<Linear>(store, linear);
                m_graph.mapper.connect<User>(store, user);
            }

            /**
             * @brief Translate `T_Store_Tiled` to `StoreTiled`.
             *
             * Coordinates for `T_Store_Tiled` are:
             *
             *                DestructMacroTile                         Join
             *     MacroTile ------------------> { SubDimension, ... } -----> User
             *
             */
            void operator()(Operations::T_Store_Tiled const& tstore)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Store_Tiled");
                std::ostringstream msg;
                for(auto const& [a, b] : m_op)
                    msg << a << "->" << b << std::endl;
                AssertFatal(m_op.count(tstore.getSrcTag()) > 0,
                            "Unknown command tag",
                            ShowValue(tstore.getSrcTag()),
                            msg.str());

                auto tensor = m_command->getOperation<Operations::Tensor>(tstore.getDstTag());

                std::vector<int> dims;
                auto const       strides        = tensor.strides();
                auto const       literalStrides = tensor.literalStrides();
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    std::shared_ptr<Expression::Expression> strideExpr;
                    if(literalStrides.size() > i && literalStrides[i] > 0)
                    {
                        strideExpr = Expression::literal(literalStrides[i]);
                    }
                    else
                    {
                        strideExpr = std::make_shared<Expression::Expression>(strides[i]);
                    }

                    auto dim = m_graph.coordinates.addElement(SubDimension(i, nullptr, strideExpr));
                    dims.push_back(dim);
                }

                auto tile = m_dim.at(tstore.getSrcTag());
                auto user = m_graph.coordinates.addElement(
                    User(tstore.getSrcTag(),
                         tensor.data()->name(),
                         std::make_shared<Expression::Expression>(tensor.limit())));

                m_graph.coordinates.addElement(DestructMacroTile(), std::vector<int>{tile}, dims);
                m_graph.coordinates.addElement(Join(), dims, std::vector<int>{user});
                m_graph.coordinates.addElement(DataFlow(), {tile}, {user});

                auto store = m_graph.control.addElement(StoreTiled(tensor.dataType()));
                auto last  = m_op.at(tstore.getSrcTag());
                m_graph.control.addElement(Sequence(), {last}, {store});

                m_graph.mapper.connect<MacroTile>(store, tile);
                m_graph.mapper.connect<User>(store, user);
            }

            /**
             * @brief Translate `T_Execute` to element operations.
             *
             * Each element operation becomes a node in the control
             * graph.  New output nodes are added to the coordinate
             * graph.  Input and output coordinates are connected with
             * `DataFlow` edges.
             */
            void operator()(Operations::T_Execute const& exec)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Execute");

                for(auto const& xop : exec.getXOps())
                {
                    auto sinputs = std::visit(
                        overloaded{
                            [](Operations::E_Ternary const& op) {
                                return std::vector<Operations::OperationTag>{op.a, op.b, op.c};
                            },
                            [](Operations::E_Binary const& op) {
                                return std::vector<Operations::OperationTag>{op.a, op.b};
                            },
                            [](Operations::E_Unary const& op) {
                                return std::vector<Operations::OperationTag>{op.a};
                            },
                            [](Operations::Nop const& op) {
                                return std::vector<Operations::OperationTag>{};
                            },
                        },
                        *xop);

                    Operations::Outputs outputsVisitor;
                    auto                soutputs = outputsVisitor.call(*xop);

                    std::vector<int> coordinate_inputs, coordinate_outputs;
                    std::vector<int> control_inputs;

                    for(auto const& sinput : sinputs)
                    {
                        AssertFatal(m_op.count(sinput) > 0 || m_expr.count(sinput) > 0,
                                    "XOp input needs to be a literal or have a node in kernel "
                                    "control graph.");
                        AssertFatal(m_dim.count(sinput) > 0 || m_expr.count(sinput) > 0,
                                    "XOp input needs to be a literal or have a dimension in kernel "
                                    "coordinate graph.");
                        if(m_op.count(sinput) > 0)
                            control_inputs.push_back(m_op.at(sinput));
                        if(m_dim.count(sinput) > 0)
                            coordinate_inputs.push_back(m_dim.at(sinput));
                    }

                    auto cTag = std::visit(
                        [](auto const& a) -> Operations::OperationTag { return a.getTag(); }, *xop);
                    auto dimension
                        = promoteDimensions(cTag, m_graph.coordinates, coordinate_inputs);
                    for(auto const& soutput : soutputs)
                    {
                        AssertFatal(m_op.count(soutput) == 0,
                                    "XOp output already exists in kernel graph.");
                        AssertFatal(m_dim.count(soutput) == 0,
                                    "XOp output already exists in kernel graph.");
                        auto tag = m_graph.coordinates.addElement(dimension);
                        coordinate_outputs.push_back(tag);
                        m_dim.insert_or_assign(soutput, tag);
                    }

                    m_graph.coordinates.addElement(
                        DataFlow(), coordinate_inputs, coordinate_outputs);

                    // Translate coordinate input tags to DataFlowTag expressions.  Each DataFlowTag
                    // refers to the corresponding coordinate input tag, and uses DataType::None to
                    // represent a "deferred" type.
                    std::vector<Expression::ExpressionPtr> dflow(sinputs.size());
                    std::transform(sinputs.cbegin(),
                                   sinputs.cend(),
                                   dflow.begin(),
                                   [=, this](Operations::OperationTag sinput) {
                                       if(m_dim.count(sinput) > 0)
                                       {
                                           return std::make_shared<Expression::Expression>(
                                               Expression::DataFlowTag{m_dim.at(sinput),
                                                                       Register::Type::Vector,
                                                                       DataType::None});
                                       }
                                       else
                                       {
                                           return m_expr.at(sinput);
                                       }
                                   });

                    // Translate XOp to an Expression
                    auto expr = std::visit(
                        rocRoller::overloaded{
                            [&](Operations::E_Neg const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Negate{dflow[0]});
                            },
                            [](Operations::E_Abs const& op) -> Expression::ExpressionPtr {
                                // TODO Implement E_Abs XOp
                                Throw<FatalError>("T_Execute E_Abs not implemented yet.");
                            },
                            [](Operations::E_Not const& op) -> Expression::ExpressionPtr {
                                // TODO Implement E_Not XOp
                                Throw<FatalError>("T_Execute E_Not not implemented yet.");
                            },
                            [&](Operations::E_Cvt const& op) -> Expression::ExpressionPtr {
                                switch(op.destType)
                                {
                                case DataType::Float:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::Convert{{.arg{dflow[0]}}, DataType::Float});
                                case DataType::Half:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::Convert{{.arg{dflow[0]}}, DataType::Half});
                                case DataType::FP8:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::Convert{{.arg{dflow[0]}}, DataType::FP8});
                                case DataType::BF8:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::Convert{{.arg{dflow[0]}}, DataType::BF8});
                                case DataType::BFloat16:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::Convert{{.arg{dflow[0]}}, DataType::BFloat16});
                                default:
                                    Throw<FatalError>("T_Execute E_Cvt type not implemented yet.");
                                }
                            },
                            [&](Operations::E_RandomNumber const& op) -> Expression::ExpressionPtr {
                                // In kernel graph transformation (AddPRNG), a new Assign op will
                                // be inserted before each E_RandomNumber to generate and store a new
                                // random number in the seed VGPR. So, here E_RandomNumber just gets
                                // the generated random number from input (seed VGPR). To ensure a
                                // VGPR will be allocated for the output of E_RandomNumber, the data
                                // type of the input must be set to UInt32 instead of None.
                                auto& dataFlowTag
                                    = std::get<rocRoller::Expression::DataFlowTag>(*dflow[0]);
                                dataFlowTag.varType.dataType = DataType::UInt32;
                                return dflow[0];
                            },
                            [&](Operations::E_StochasticRoundingCvt const& op)
                                -> Expression::ExpressionPtr {
                                // Stochastic conversion
                                switch(op.destType)
                                {
                                case DataType::FP8:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::SRConvert<DataType::FP8>{dflow[0], dflow[1]});
                                case DataType::BF8:
                                    return std::make_shared<Expression::Expression>(
                                        Expression::SRConvert<DataType::BF8>{dflow[0], dflow[1]});
                                default:
                                    Throw<FatalError>("Not implemented yet.");
                                }
                            },
                            [&](Operations::E_Add const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Add{dflow[0], dflow[1]});
                            },
                            [&](Operations::E_Sub const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Subtract{dflow[0], dflow[1]});
                            },
                            [&](Operations::E_Mul const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Multiply{dflow[0], dflow[1]});
                            },
                            [&](Operations::E_Div const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Divide{dflow[0], dflow[1]});
                            },
                            [](Operations::E_And const& op) -> Expression::ExpressionPtr {
                                // TODO Implement E_And XOp
                                Throw<FatalError>("T_Execute E_And not implemented yet.");
                            },
                            [](Operations::E_Or const& op) -> Expression::ExpressionPtr {
                                // TODO Implement E_Or XOp
                                Throw<FatalError>("T_Execute E_Or not implemented yet.");
                            },
                            [&](Operations::E_GreaterThan const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::GreaterThan{dflow[0], dflow[1]});
                            },
                            [&](Operations::E_Conditional const& op) -> Expression::ExpressionPtr {
                                return std::make_shared<Expression::Expression>(
                                    Expression::Conditional{dflow[0], dflow[1], dflow[2]});
                            },
                        },
                        *xop);

                    auto op = m_graph.control.addElement(Assign{Register::Type::Vector, expr});

                    for(auto const& input : control_inputs)
                    {
                        m_graph.control.addElement(Sequence(), {input}, {op});
                    }

                    AssertFatal(coordinate_outputs.size() == 1,
                                "Element op must have a single output.");
                    m_graph.mapper.connect(op, coordinate_outputs[0], NaryArgument::DEST);

                    Operations::TagVisitor tagVisitor;
                    m_op.insert_or_assign(tagVisitor.call(*xop), op);
                }
            }

            void operator()(Operations::RandomNumberGenerator const& rng)
            {
                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::TranslateVisitor::RandomNumberGenerator");

                // Coordinate Graph
                auto& coordinates = m_graph.coordinates;

                auto seedVGPR = coordinates.addElement(VGPR(rng.getTag()));
                m_dim.insert_or_assign(rng.getTag(), seedVGPR);
                AssertFatal(m_dim.contains(rng.seed),
                            "A seed should be provided to be the initial seed value of random "
                            "number generator");
                coordinates.addElement(DataFlow(), {m_dim.at(rng.seed)}, {seedVGPR});

                // Control Graph
                auto& control = m_graph.control;

                bool const addTID
                    = rng.getSeedMode() == Operations::RandomNumberGenerator::SeedMode::PerThread;
                auto seedPRNG = control.addElement(SeedPRNG{addTID});
                m_op.insert_or_assign(rng.getTag(), seedPRNG);
                control.addElement(Sequence(), {m_op.at(rng.seed)}, {seedPRNG});

                // Associate SeedPRNG op with seed VGPR (DEST) and the user-given seed value (RHS)
                m_graph.mapper.connect(seedPRNG, seedVGPR, NaryArgument::DEST);
                m_graph.mapper.connect(seedPRNG, m_dim.at(rng.seed), NaryArgument::RHS);
            }

            /**
             * @brief Translate `T_Mul` to `TensorContraction`.
             *
             * Macro tiles in the coordinate graph are connected with
             * a `DataFlow` edge.
             */
            void operator()(Operations::T_Mul const& mul)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Mul");

                auto D = m_graph.coordinates.addElement(MacroTile());

                m_dim[mul.getTag()] = D;

                auto aSource = m_command->findTag(mul.a);
                auto bSource = m_command->findTag(mul.b);

                auto getBlockParams = rocRoller::overloaded{
                    [&](auto const& op) {
                        return std::make_tuple(Operations::ScaleMode::None, std::vector<size_t>());
                    },
                    [&](Operations::BlockScale const& op) {
                        return std::make_tuple(op.scaleMode(), op.strides());
                    }};

                TensorContraction contraction({1}, {0}, mul.accType);
                std::tie(contraction.scaleModeA, contraction.scaleStridesA)
                    = std::visit(getBlockParams, *aSource);
                std::tie(contraction.scaleModeB, contraction.scaleStridesB)
                    = std::visit(getBlockParams, *bSource);

                // contraction dims are {1} and {0}, which is matrix multiplication
                auto TC            = m_graph.control.addElement(NOP{});
                m_op[mul.getTag()] = TC;
                m_graph.mapper.connect(TC, D, NaryArgument::DEST);

                std::vector<int> sourceDims;

                auto connectBlockScale = [&](Operations::BlockScale const& op,
                                             Operations::OperationTag      inputTag,
                                             NaryArgument                  valueArg,
                                             NaryArgument scaleArg) -> std::vector<size_t> {
                    auto mode = op.scaleMode();
                    AssertFatal(mode != Operations::ScaleMode::Inline, ShowValue(mode));

                    auto scaleInput   = op.scale().value();
                    auto scaleInputOp = m_command->findTag(scaleInput);
                    AssertFatal(scaleInputOp != nullptr);

                    using TensorAndTranspose
                        = std::tuple<Operations::OperationTag, std::vector<size_t>>;
                    auto getTensorAndTranspose = rocRoller::overloaded{
                        [](Operations::SubTileTranspose const& op) -> TensorAndTranspose {
                            return {op.input(), op.tileDimensions()};
                        },
                        [](Operations::Nop const& op) -> TensorAndTranspose {
                            return {Operations::OperationTag(), {}};
                        },
                        [](auto const& op) -> TensorAndTranspose {
                            return {op.getTag(), {}};
                        }};

                    auto [scaleTensor, scaleTranspose]
                        = std::visit(getTensorAndTranspose, *scaleInputOp);

                    if(!scaleTranspose.empty())
                    {
                        AssertFatal(m_params);
                        auto info = m_params->getDimensionInfo().at(scaleTensor);
                        auto tile
                            = std::get<rocRoller::KernelGraph::CoordinateGraph::MacroTile>(info);
                        size_t miKScale = tile.miTileSizes.at(2);

                        std::vector<size_t> expectedTile{64, 4, miKScale};

                        AssertFatal(scaleTranspose == expectedTile,
                                    ShowValue(scaleTranspose),
                                    ShowValue(expectedTile),
                                    ShowValue(valueArg),
                                    ShowValue(tile),
                                    ShowValue(tile.miTileSizes));
                    }

                    auto X      = m_dim.at(op.data());
                    auto XScale = m_dim.at(scaleTensor);

                    auto loadX      = m_op.at(op.data());
                    auto loadXScale = m_op.at(scaleTensor);

                    m_graph.control.addElement(Sequence(), {loadX}, {TC});
                    m_graph.control.addElement(Sequence(), {loadXScale}, {TC});

                    m_graph.mapper.connect(TC, X, valueArg);
                    m_graph.mapper.connect(TC, XScale, scaleArg);

                    sourceDims.insert(sourceDims.end(), {X, XScale});

                    return scaleTranspose;
                };

                auto handleInput = rocRoller::overloaded{
                    [&](auto const& op, auto, auto, auto) -> std::vector<size_t> {
                        Throw<FatalError>("Can't go here!");
                        return {};
                    },

                    [&](Operations::T_Load_Tiled const& op,
                        Operations::OperationTag        inputTag,
                        NaryArgument                    valueArg,
                        auto) -> std::vector<size_t> {
                        auto AB     = m_dim.at(inputTag);
                        auto loadAB = m_op.at(inputTag);

                        sourceDims.push_back(AB);

                        m_graph.control.addElement(Sequence(), {loadAB}, {TC});

                        m_graph.mapper.connect(TC, AB, valueArg);

                        return {};
                    },
                    connectBlockScale};

                // Handle A, either T_Load_Tiled or BlockScale
                contraction.scalePreShuffledTileA
                    = std::visit(handleInput,
                                 *aSource,
                                 singleVariant(mul.a),
                                 singleVariant(NaryArgument::LHS),
                                 singleVariant(NaryArgument::LHS_SCALE));

                contraction.scalePreShuffledTileB
                    = std::visit(handleInput,
                                 *bSource,
                                 singleVariant(mul.b),
                                 singleVariant(NaryArgument::RHS),
                                 singleVariant(NaryArgument::RHS_SCALE));

                m_graph.coordinates.addElement(DataFlow(), sourceDims, std::vector{D});

                // Replace contraction in graph after we have filled all the fields.
                m_graph.control.setElement(TC, std::move(contraction));
            }

            void operator()(Operations::BlockScale const& t)
            {
                AssertFatal(t.scaleMode() != Operations::ScaleMode::Inline,
                            "ScaleMode::Inline not supported yet.");
            }

            void operator()(Operations::SubTileTranspose const& t) {}

            void operator()(Operations::Literal const& literal)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::Literal");

                auto expr = Expression::literal(literal.value());
                m_expr.insert_or_assign(literal.getTag(), expr);
            }

            void operator()(Operations::Nop const& x) {}

            void operator()(Operations::Tensor const& t) {}
            void operator()(Operations::Scalar const& t) {}

            KernelGraph call(CommandPtr command, CommandParametersPtr params)
            {
                m_command = command;
                m_params  = params;
                for(auto const& op : command->operations())
                {
                    std::visit(*this, *op);
                }
                return m_graph;
            }

        private:
            KernelGraph m_graph;

            // root/kernel tag
            int m_kernel;

            // command tag -> operation tag
            std::map<Operations::OperationTag, int> m_op;

            // literal command tag -> Expression
            std::map<Operations::OperationTag, Expression::ExpressionPtr> m_expr;

            // command tag -> dimension/coordinate tag
            std::map<Operations::OperationTag, int> m_dim;

            CommandPtr           m_command;
            CommandParametersPtr m_params;
        };

        KernelGraph translate(CommandPtr command, CommandParametersPtr params)
        {
            TIMER(t, "KernelGraph::translate");
            rocRoller::Log::getLogger()->debug("KernelGraph::translate(); Command\n{}",
                                               command->toString());
            TranslateVisitor visitor;
            return visitor.call(command, params);
        }
    }
}
