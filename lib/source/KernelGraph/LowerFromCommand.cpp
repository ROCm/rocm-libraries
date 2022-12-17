/*
 * Command to KernelGraph translator
 */

#include <variant>
#include <vector>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace CoordGraph;
        using namespace ControlHypergraph;

        /**
         * @brief Promote inputs to an appropriate output.
         *
         * For example, given VGPR and Linear inputs, output should be Linear.
         */
        Dimension promoteDimensions(CoordinateHypergraph const& graph, std::vector<int> const& dims)
        {
            Dimension rv = VGPR();
            for(auto tag : dims)
            {
                auto element = graph.getElement(tag);

                visit(
                    rocRoller::overloaded{
                        [](VGPR const& op) {},
                        [&rv](Linear const& op) {
                            AssertFatal(
                                !std::holds_alternative<MacroTile>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = Linear();
                        },
                        [&rv](MacroTile const& op) {
                            AssertFatal(
                                !std::holds_alternative<Linear>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = MacroTile();
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
                m_kernel = graph.control.addElement(Kernel());
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
                // TODO: offsets and limits
                auto sizes   = tload.sizes();
                auto strides = tload.strides();

                auto total_size_expr = std::make_shared<Expression::Expression>(sizes[0]);

                auto user = graph.coordinates.addElement(User(tload.data()->name()));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    dims.push_back(
                        graph.coordinates.addElement(SubDimension(i, size_expr, stride_expr)));
                    if(i > 0)
                        total_size_expr = total_size_expr * size_expr;
                }

                graph.coordinates.addElement(Split(), std::vector<int>{user}, dims);

                auto unit_stride = Expression::literal(1u);
                auto linear = graph.coordinates.addElement(Linear(total_size_expr, unit_stride));

                graph.coordinates.addElement(Flatten(), dims, std::vector<int>{linear});
                graph.coordinates.addElement(DataFlow(), {user}, {linear});

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tload.getTag()));
                auto load  = graph.control.addElement(LoadLinear(vtype));
                graph.control.addElement(Body(), {m_kernel}, {load});

                graph.mapper.connect<User>(load, user);
                graph.mapper.connect<Linear>(load, linear);

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
                auto user = graph.coordinates.addElement(User(tload.data()->name()));
                auto vgpr = graph.coordinates.addElement(VGPR());

                graph.coordinates.addElement(DataFlow(), {user}, {vgpr});

                auto load = graph.control.addElement(LoadVGPR(tload.variableType(), true));
                graph.control.addElement(Body(), {m_kernel}, {load});

                graph.mapper.connect<User>(load, user);
                graph.mapper.connect<VGPR>(load, vgpr);

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

                // TODO: offsets and limits
                auto const tag     = tload.getTag();
                auto const sizes   = tload.sizes();
                auto const strides = tload.strides();

                auto user = graph.coordinates.addElement(User(tload.data()->name()));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    auto dim
                        = graph.coordinates.addElement(SubDimension(i, size_expr, stride_expr));
                    dims.push_back(dim);
                }

                auto tiled = graph.coordinates.addElement(MacroTile(sizes.size()));

                graph.coordinates.addElement(Split(), std::vector<int>{user}, dims);
                graph.coordinates.addElement(ConstructMacroTile(), dims, std::vector<int>{tiled});
                graph.coordinates.addElement(DataFlow(), {user}, {tiled});

                auto load = graph.control.addElement(LoadTiled(tload.variableType()));
                graph.control.addElement(Body(), {m_kernel}, {load});

                graph.mapper.connect<User>(load, user);
                graph.mapper.connect<MacroTile>(load, tiled);

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
                AssertFatal(m_op.count(tstore.getTag()) > 0,
                            "Unknown command tag",
                            ShowValue(tstore.getTag()));

                std::vector<int> dims;
                auto             strides = tstore.strides();
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim = graph.coordinates.addElement(SubDimension(i, nullptr, stride_expr));
                    dims.push_back(dim);
                }

                auto linear = m_dim.at(tstore.getTag());
                auto user   = graph.coordinates.addElement(User(tstore.data()->name()));

                graph.coordinates.addElement(Split(), std::vector<int>{linear}, dims);
                graph.coordinates.addElement(Join(), dims, std::vector<int>{user});
                graph.coordinates.addElement(DataFlow(), {linear}, {user});

                auto store = graph.control.addElement(StoreLinear());
                auto last  = m_op.at(tstore.getTag());
                graph.control.addElement(Sequence(), {last}, {store});

                graph.mapper.connect<Linear>(store, linear);
                graph.mapper.connect<User>(store, user);
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
                AssertFatal(m_op.count(tstore.getTag()) > 0,
                            "Unknown command tag",
                            ShowValue(tstore.getTag()));

                std::vector<int> dims;
                auto             strides = tstore.strides();
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim = graph.coordinates.addElement(SubDimension(i, nullptr, stride_expr));
                    dims.push_back(dim);
                }

                auto tile = m_dim.at(tstore.getTag());
                auto user = graph.coordinates.addElement(User(tstore.data()->name()));

                graph.coordinates.addElement(DestructMacroTile(), std::vector<int>{tile}, dims);
                graph.coordinates.addElement(Join(), dims, std::vector<int>{user});
                graph.coordinates.addElement(DataFlow(), {tile}, {user});

                auto store = graph.control.addElement(StoreTiled(tstore.dataType()));
                auto last  = m_op.at(tstore.getTag());
                graph.control.addElement(Sequence(), {last}, {store});

                graph.mapper.connect<MacroTile>(store, tile);
                graph.mapper.connect<User>(store, user);
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
                for(auto const& xop : exec.getXOps())
                {
                    auto sinputs = std::visit(
                        overloaded{
                            [&](Operations::E_Binary const& op) {
                                return std::vector<int>{op.a, op.b};
                            },
                            [&](Operations::E_Unary const& op) { return std::vector<int>{op.a}; },
                            [&](Operations::Nop const& op) { return std::vector<int>{}; },
                        },
                        *xop);

                    auto soutputs = Operations::Outputs()(*xop);

                    std::vector<int> coordinate_inputs, coordinate_outputs;
                    std::vector<int> control_inputs;

                    for(auto const& sinput : sinputs)
                    {
                        AssertFatal(m_op.count(sinput) > 0,
                                    "Unable to find XOp inputs in kernel control graph.");
                        AssertFatal(m_dim.count(sinput) > 0,
                                    "Unable to find XOp inputs in kernel coordinate graph.");
                        control_inputs.push_back(m_op.at(sinput));
                        coordinate_inputs.push_back(m_dim.at(sinput));
                    }

                    auto coordinateType = promoteDimensions(graph.coordinates, coordinate_inputs);
                    for(auto const& soutput : soutputs)
                    {
                        AssertFatal(m_op.count(soutput) == 0,
                                    "XOp output already exists in kernel graph.");
                        AssertFatal(m_dim.count(soutput) == 0,
                                    "XOp output already exists in kernel graph.");
                        auto dimension = graph.coordinates.addElement(coordinateType);
                        coordinate_outputs.push_back(dimension);
                        m_dim.insert_or_assign(soutput, dimension);
                    }

                    graph.coordinates.addElement(DataFlow(), coordinate_inputs, coordinate_outputs);

                    auto op = graph.control.addElement(
                        ElementOp(*xop,
                                  coordinate_inputs[0],
                                  coordinate_inputs.size() > 1 ? coordinate_inputs[1] : -1));

                    for(auto const& input : control_inputs)
                    {
                        graph.control.addElement(Sequence(), {input}, {op});
                    }

                    AssertFatal(coordinate_outputs.size() == 1,
                                "Element op must have a single output.");
                    auto odim
                        = std::get<Dimension>(graph.coordinates.getElement(coordinate_outputs[0]));
                    if(isDimension<MacroTile>(odim))
                        graph.mapper.connect<MacroTile>(op, coordinate_outputs[0]);
                    else
                        graph.mapper.connect<Linear>(op, coordinate_outputs[0]);

                    m_op.insert_or_assign(Operations::Tag()(*xop), op);
                }
            }

            /**
             * @brief Translate `T_Mul` to `TensorContraction`.
             *
             * Macro tiles in the coorindate graph are connected with
             * a `DataFlow` edge.
             */
            void operator()(Operations::T_Mul const& mul)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Mul");

                auto A = m_dim.at(mul.a);
                auto B = m_dim.at(mul.b);
                auto D = graph.coordinates.addElement(MacroTile());
                m_dim.insert_or_assign(mul.dest, D);

                graph.coordinates.addElement(DataFlow(), {A, B}, {D});

                // contraction dims are {1} and {0}, which is matrix multiplication
                auto TC = graph.control.addElement(TensorContraction(A, B, {1}, {0}));
                m_op.insert_or_assign(mul.dest, TC);

                auto loadA = m_op.at(mul.a);
                auto loadB = m_op.at(mul.b);

                graph.control.addElement(Sequence(), {loadA}, {TC});
                graph.control.addElement(Sequence(), {loadB}, {TC});

                graph.mapper.connect<MacroTile>(TC, D);

#ifndef NDEBUG
                auto parents = graph.control.parentNodes(TC).to<std::vector>();
                AssertFatal(parents.size() == 2, "T_MUL requires 2 inputs.");
                for(auto const& parent : parents)
                {
                    auto element = graph.control.getElement(parent);
                    auto node    = std::get<Operation>(graph.control.getElement(parent));
                    AssertFatal(std::holds_alternative<LoadTiled>(node),
                                "T_MUL inputs must be tiled.");
                }
#endif
            }

            void operator()(Operations::Nop const& x) {}

            KernelHypergraph operator()(std::shared_ptr<Command> command)
            {
                m_command = command;
                for(auto const& op : command->operations())
                {
                    std::visit(*this, *op);
                }
                return graph;
            }

        private:
            KernelHypergraph graph;

            // root/kernel tag
            int m_kernel;

            // command tag -> operation tag
            std::map<int, int> m_op;

            // command tag -> dimension/coordinate tag
            std::map<int, int> m_dim;

            std::shared_ptr<Command> m_command;
        };

        KernelHypergraph translate(std::shared_ptr<Command> command)
        {
            TIMER(t, "KernelGraph::translate");
            rocRoller::Log::getLogger()->debug("KernelGraph::translate(); Command\n{}",
                                               command->toString());
            return TranslateVisitor()(command);
        }
    }
}
