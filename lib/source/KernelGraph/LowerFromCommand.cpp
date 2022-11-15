#include <variant>
#include <vector>

#include "KernelGraph/ControlHypergraph/ControlEdge.hpp"
#include "KernelGraph/ControlHypergraph/Operation.hpp"
#include "KernelGraph/ControlHypergraph/Operation_fwd.hpp"
#include "KernelGraph/CoordGraph/Dimension.hpp"
#include "KernelGraph/CoordinateTransform/Dimension.hpp"
#include "Operations/T_Execute.hpp"
#include "Utilities/Error.hpp"
#include "Utilities/Utils.hpp"
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

#include <rocRoller/KernelGraph/ControlHypergraph/ControlHypergraph.hpp>
#include <rocRoller/KernelGraph/CoordGraph/CoordinateHypergraph.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Edge.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        namespace Expression = rocRoller::Expression;

        /***********************************
         * Command to HyperGraph translator
         */

        // Delete this when graph rearch complete
        struct MakeOutputVisitor
        {
            int ndtag;

            CoordinateTransform::Dimension operator()(CoordinateTransform::MacroTile const& a)
            {
                return CoordinateTransform::MacroTile(ndtag);
            }

            template <typename T>
            CoordinateTransform::Dimension operator()(T const& a)
            {
                return CoordinateTransform::Linear(ndtag);
            }
        };

        // Delete this when graph rearch complete
        struct TranslateVisitor
        {
            KernelGraph m_kgraph;

            /*
             * Command operations
             */

            /* T_Load_Linear becomes:
             *
             *       Split                         LoadLinear
             * User ------> { SubDimension, ... } -----------> Linear
             *
             * For example:
             *
             *   T_Load_Linear[dataflow=1, sizes={16, 32}, strides={32, 1}]
             *
             * becomes
             *          Split
             * User(1) ------> { SubDimension({1, 0}, size=16, stride=32),
             *                   SubDimension({1, 1}, size=32, stride=1) }
             *
             *          Flatten
             *         --------> Linear(1)
             *
             * and
             *
             *          LoadLinear
             * User(1) -----------> Linear(1)
             *
             */
            void operator()(Operations::T_Load_Linear const& load)
            {
                // TODO: offsets and limits
                auto sizes   = load.sizes();
                auto strides = load.strides();

                auto total_size_expr = std::make_shared<Expression::Expression>(sizes[0]);

                std::vector<CoordinateTransform::Dimension> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    auto dim = CoordinateTransform::SubDimension(
                        load.getTag(), i, size_expr, stride_expr);
                    dims.push_back(dim);
                    if(i > 0)
                        total_size_expr = total_size_expr * size_expr;
                }

                auto user = CoordinateTransform::User(load.getTag(), load.data()->name());
                m_kgraph.coordinates.addEdge({user}, dims, CoordinateTransform::Split());

                auto unit_stride = Expression::literal(1u);
                auto linear
                    = CoordinateTransform::Linear(load.getTag(), total_size_expr, unit_stride);

                m_kgraph.coordinates.addEdge(dims, {linear}, CoordinateTransform::Flatten());
                m_kgraph.coordinates.addEdge({user}, {linear}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge({ControlGraph::Kernel()},
                                         {ControlGraph::LoadLinear(load.getTag())},
                                         ControlGraph::Body());
            }

            /* T_Load_Scalar becomes:
             *
             *       LoadVGPR
             * User ---------> VGPR
             *
             */
            void operator()(Operations::T_Load_Scalar const& load)
            {
                auto user = CoordinateTransform::User(load.getTag(), load.data()->name());
                auto vgpr = CoordinateTransform::VGPR(load.getTag());

                m_kgraph.coordinates.addEdge({user}, {vgpr}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge({ControlGraph::Kernel()},
                                         {ControlGraph::LoadVGPR(load.getTag())},
                                         ControlGraph::Body());
            }

            /* T_Load_Tiled becomes:
             *
             *       Split                         ConstructMacroTile
             * User ------> { SubDimension, ... } -------------------> MacroTile
             *
             * For example:
             *
             *   T_Load_Tiled[dataflow=1, sizes={16, 32}, strides={32, 1}]
             *
             * becomes
             *          Split
             * User(1) ------> { SubDimension({1, 0}, size=16, stride=32),
             *                   SubDimension({1, 1}, size=32, stride=1) }
             *
             *          ConstructMacroTile
             *         -------------------> MacroTile(1)
             *
             * and
             *
             *          LoadTile
             * User(1) ---------> MacroTile(1)
             *
             * The sizes, storage location, and affinity of the tile
             * do not need to be known at this stage.
             */
            void operator()(Operations::T_Load_Tiled const& load)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Load_Tiled");

                // TODO: offsets and limits
                auto const tag     = load.getTag();
                auto const sizes   = load.sizes();
                auto const strides = load.strides();

                std::vector<CoordinateTransform::Dimension> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    auto dim = CoordinateTransform::SubDimension(tag, i, size_expr, stride_expr);
                    dims.push_back(dim);
                }

                auto user  = CoordinateTransform::User(tag, load.data()->name());
                auto tiled = CoordinateTransform::MacroTile(tag, sizes.size());

                m_kgraph.coordinates.addEdge({user}, dims, CoordinateTransform::Split());
                m_kgraph.coordinates.addEdge(
                    dims, {tiled}, CoordinateTransform::ConstructMacroTile());
                m_kgraph.coordinates.addEdge({user}, {tiled}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge(
                    {ControlGraph::Kernel()}, {ControlGraph::LoadTiled(tag)}, ControlGraph::Body());
            }

            /* T_Store_Linear becomes:
             *
             *            MakeOutput           Split                          Join
             *    Linear -----------> LinearO ------> { SubDimensionO, ... } -----> UserO
             *
             * and
             *
             *            StoreLinear
             *    Linear ------------> User
             *
             * The 'MakeOutput' edge connects the Linear dimension to
             * the output Linear diemsion; and serves to break cycles
             * when copying inputs to outputs.
             */
            void operator()(Operations::T_Store_Linear const& store)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Store_Linear");
                auto strides = store.strides();

                std::vector<CoordinateTransform::Dimension> dims;
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim         = CoordinateTransform::SubDimension(
                        store.getTag(), i, nullptr, stride_expr, true);
                    dims.push_back(dim);
                }

                auto inLinear  = CoordinateTransform::Linear(store.getTag());
                auto outLinear = CoordinateTransform::Linear(store.getTag(), true);
                auto user = CoordinateTransform::User(store.getTag(), store.data()->name(), true);

                m_kgraph.coordinates.addEdge(
                    {inLinear}, {outLinear}, CoordinateTransform::MakeOutput());
                m_kgraph.coordinates.addEdge({outLinear}, dims, CoordinateTransform::Split());
                m_kgraph.coordinates.addEdge(dims, {user}, CoordinateTransform::Join());
                m_kgraph.coordinates.addEdge({inLinear}, {user}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge({user}, {ControlGraph::StoreLinear(store.getTag())});
            }

            /* T_Store_Tiled becomes:
             *
             *           MakeOutput               DestructMacroTile                         Join
             * MacroTile ----------> MacroTiledO ------------------> { SubDimension, ... } -----> User
             *
             * and
             *
             *            StoreTile
             * MacroTile ----------> User
             */
            void operator()(Operations::T_Store_Tiled const& store)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Store_Tiled");

                auto strides = store.strides();

                std::vector<CoordinateTransform::Dimension> dims;
                for(size_t i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim         = CoordinateTransform::SubDimension(
                        store.getTag(), i, nullptr, stride_expr, true);
                    dims.push_back(dim);
                }

                auto inTile  = CoordinateTransform::MacroTile(store.getTag(), dims.size());
                auto outTile = CoordinateTransform::MacroTile(store.getTag(), dims.size(), true);
                auto user = CoordinateTransform::User(store.getTag(), store.data()->name(), true);

                m_kgraph.coordinates.addEdge(
                    {inTile}, {outTile}, CoordinateTransform::MakeOutput());
                m_kgraph.coordinates.addEdge(
                    {outTile}, dims, CoordinateTransform::DestructMacroTile());
                m_kgraph.coordinates.addEdge(dims, {user}, CoordinateTransform::Join());
                m_kgraph.coordinates.addEdge({outTile}, {user}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge(
                    {user}, {ControlGraph::StoreTiled(store.getTag(), store.dataType())});
            }

            /*
             * T_Execute: each XOp becomes
             *
             *          XOp
             *  Linear ----> Linear
             */
            void operator()(Operations::T_Execute const& exec)
            {
                for(auto const& xop : exec.getXOps())
                {
                    auto sinputs = Operations::Inputs()(*xop);
                    auto inputs  = m_kgraph.coordinates.getLinearDimensions(sinputs);
                    AssertRecoverable(inputs.size() == sinputs.size(),
                                      "Unable to find XOp inputs in kernel graph.");

                    auto soutputs = Operations::Outputs()(*xop);
                    auto outputs  = m_kgraph.coordinates.getLinearDimensions(soutputs);
                    AssertRecoverable(outputs.empty(),
                                      "XOp output already exists in kernel graph.");

                    for(auto const& ndtag : soutputs)
                    {
                        MakeOutputVisitor mkoutput{ndtag};
                        outputs.push_back(std::visit(mkoutput, inputs[0]));
                    }

                    m_kgraph.coordinates.addEdge(inputs, outputs, CoordinateTransform::DataFlow());

                    // TODO: make sure XOp is elemental
                    m_kgraph.control.addEdge(
                        inputs, {ControlGraph::ElementOp(Operations::Tag()(*xop), xop)});
                }
            }

            /*
             * T_Mul becomes
             *
             *                          TensorContraction
             *  {MacroTile, MacroTile} ------------------> MacroTile
             */
            void operator()(Operations::T_Mul const& mul)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::TranslateVisitor::T_Mul");

                auto A = MacroTile(mul.a);
                auto B = MacroTile(mul.b);
                auto D = MacroTile(mul.dest);

                m_kgraph.coordinates.addEdge({A, B}, {D}, CoordinateTransform::DataFlow());

                // contraction dims are {1} and {0}, which is matrix multiplication
                auto TC = ControlGraph::TensorContraction(mul.getTag(), A, B, {1}, {0});

                auto loadA = ControlGraph::LoadTiled(A.tag);
                auto loadB = ControlGraph::LoadTiled(B.tag);

                m_kgraph.control.addEdge({loadA, loadB}, {TC}, ControlGraph::Sequence());
#ifndef NDEBUG
                auto parents = m_kgraph.control.getInputs(getTag(TC));
                AssertFatal(parents.size() == 2, "T_MUL requires 2 inputs.");
                for(auto const& parent : parents)
                {
                    AssertFatal(std::holds_alternative<ControlGraph::LoadTiled>(parent),
                                "T_MUL inputs must be tiled.");
                }
#endif
            }

            /*
             * Nops... don't do anything
             */
            void operator()(Operations::Nop const& x) {}

            /*
             * Dispatch!
             */
            KernelGraph operator()(std::shared_ptr<Command> const command)
            {
                for(auto const& op : command->operations())
                {
                    std::visit(*this, *op);
                }
                return m_kgraph;
            }
        };

        // Delete this when graph rearch complete
        KernelGraph translate(std::shared_ptr<Command> command)
        {
            TIMER(t, "KernelGraph::translate");
            rocRoller::Log::getLogger()->debug("KernelGraph::translate(); Command\n{}",
                                               command->toString());
            TranslateVisitor visitor;
            return visitor(command);
        }

        /**
         * Promote element operation (E_Mul, E_Add etc) inputs to an appropriate output.
         *
         * For example, given VGPR and Linear inputs, output should be Linear.
         */
        CoordGraph::Dimension promoteDimensions(CoordGraph::CoordinateHypergraph const& graph,
                                                std::vector<int> const&                 dims)
        {
            CoordGraph::Dimension rv = CoordGraph::VGPR();
            for(auto tag : dims)
            {
                auto element = graph.getElement(tag);

                visit(
                    rocRoller::overloaded{
                        [](CoordGraph::VGPR const& op) {},
                        [&rv](CoordGraph::Linear const& op) {
                            AssertFatal(
                                !std::holds_alternative<CoordGraph::MacroTile>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = CoordGraph::Linear();
                        },
                        [&rv](CoordGraph::MacroTile const& op) {
                            AssertFatal(
                                !std::holds_alternative<CoordGraph::Linear>(rv),
                                "Element operation between Linear and MacroTile dimensions is not "
                                "well-posed.");
                            rv = CoordGraph::MacroTile();
                        },
                        [](auto const& op) {
                            Throw<FatalError>("Invalid argument of element operation.");
                        }},
                    std::get<CoordGraph::Dimension>(element));
            }
            return rv;
        }

        // Rename this when graph rearch complete
        struct TranslateVisitor2
        {
            TranslateVisitor2()
            {
                m_kernel = graph.control.addElement(ControlHypergraph::Kernel());
            }

            /*
             * Command operations
             */

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

                auto user = graph.coordinates.addElement(CoordGraph::User(tload.data()->name()));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    dims.push_back(graph.coordinates.addElement(
                        CoordGraph::SubDimension(i, size_expr, stride_expr)));
                    if(i > 0)
                        total_size_expr = total_size_expr * size_expr;
                }

                graph.coordinates.addElement(CoordGraph::Split(), std::vector<int>{user}, dims);

                auto unit_stride = Expression::literal(1u);
                auto linear      = graph.coordinates.addElement(
                    CoordGraph::Linear(total_size_expr, unit_stride));

                graph.coordinates.addElement(CoordGraph::Flatten(), dims, std::vector<int>{linear});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {user}, {linear});

                auto vtype = Operations::VariableTypeVisitor()(*m_command->findTag(tload.getTag()));
                auto load  = graph.control.addElement(ControlHypergraph::LoadLinear(vtype));
                graph.control.addElement(ControlHypergraph::Body(), {m_kernel}, {load});

                graph.mapper.connect<CoordGraph::User>(load, user);
                graph.mapper.connect<CoordGraph::Linear>(load, linear);

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
                auto user = graph.coordinates.addElement(CoordGraph::User(tload.data()->name()));
                auto vgpr = graph.coordinates.addElement(CoordGraph::VGPR());

                graph.coordinates.addElement(CoordGraph::DataFlow(), {user}, {vgpr});

                auto load = graph.control.addElement(
                    ControlHypergraph::LoadVGPR(tload.variableType(), true));
                graph.control.addElement(ControlHypergraph::Body(), {m_kernel}, {load});

                graph.mapper.connect<CoordGraph::User>(load, user);
                graph.mapper.connect<CoordGraph::VGPR>(load, vgpr);

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

                auto user = graph.coordinates.addElement(CoordGraph::User(tload.data()->name()));

                std::vector<int> dims;
                for(size_t i = 0; i < sizes.size(); ++i)
                {
                    auto size_expr   = std::make_shared<Expression::Expression>(sizes[i]);
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);

                    auto dim = graph.coordinates.addElement(
                        CoordGraph::SubDimension(i, size_expr, stride_expr));
                    dims.push_back(dim);
                }

                auto tiled = graph.coordinates.addElement(CoordGraph::MacroTile(sizes.size()));

                graph.coordinates.addElement(CoordGraph::Split(), std::vector<int>{user}, dims);
                graph.coordinates.addElement(
                    CoordGraph::ConstructMacroTile(), dims, std::vector<int>{tiled});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {user}, {tiled});

                auto load
                    = graph.control.addElement(ControlHypergraph::LoadTiled(tload.variableType()));
                graph.control.addElement(ControlHypergraph::Body(), {m_kernel}, {load});

                graph.mapper.connect<CoordGraph::User>(load, user);
                graph.mapper.connect<CoordGraph::MacroTile>(load, tiled);

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
                    auto dim         = graph.coordinates.addElement(
                        CoordGraph::SubDimension(i, nullptr, stride_expr));
                    dims.push_back(dim);
                }

                auto linear = m_dim.at(tstore.getTag());
                auto user   = graph.coordinates.addElement(CoordGraph::User(tstore.data()->name()));

                graph.coordinates.addElement(CoordGraph::Split(), std::vector<int>{linear}, dims);
                graph.coordinates.addElement(CoordGraph::Join(), dims, std::vector<int>{user});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {linear}, {user});

                auto store = graph.control.addElement(ControlHypergraph::StoreLinear());
                auto last  = m_op.at(tstore.getTag());
                graph.control.addElement(ControlHypergraph::Sequence(), {last}, {store});

                graph.mapper.connect<CoordGraph::Linear>(store, linear);
                graph.mapper.connect<CoordGraph::User>(store, user);
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
                    auto dim         = graph.coordinates.addElement(
                        CoordGraph::SubDimension(i, nullptr, stride_expr));
                    dims.push_back(dim);
                }

                auto tile = m_dim.at(tstore.getTag());
                auto user = graph.coordinates.addElement(CoordGraph::User(tstore.data()->name()));

                graph.coordinates.addElement(
                    CoordGraph::DestructMacroTile(), std::vector<int>{tile}, dims);
                graph.coordinates.addElement(CoordGraph::Join(), dims, std::vector<int>{user});
                graph.coordinates.addElement(CoordGraph::DataFlow(), {tile}, {user});

                auto store
                    = graph.control.addElement(ControlHypergraph::StoreTiled(tstore.dataType()));
                auto last = m_op.at(tstore.getTag());
                graph.control.addElement(ControlHypergraph::Sequence(), {last}, {store});

                graph.mapper.connect<CoordGraph::MacroTile>(store, tile);
                graph.mapper.connect<CoordGraph::User>(store, user);
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

                    graph.coordinates.addElement(
                        CoordGraph::DataFlow(), coordinate_inputs, coordinate_outputs);

                    auto op = graph.control.addElement(ControlHypergraph::ElementOp(
                        *xop,
                        coordinate_inputs[0],
                        coordinate_inputs.size() > 1 ? coordinate_inputs[1] : -1));

                    for(auto const& input : control_inputs)
                    {
                        graph.control.addElement(ControlHypergraph::Sequence(), {input}, {op});
                    }

                    AssertFatal(coordinate_outputs.size() == 1,
                                "Element op must have a single output.");
                    graph.mapper.connect<CoordGraph::Linear>(op, coordinate_outputs[0]);

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
                auto D = graph.coordinates.addElement(CoordGraph::MacroTile());
                m_dim.insert_or_assign(mul.dest, D);

                graph.coordinates.addElement(CoordGraph::DataFlow(), {A, B}, {D});

                // contraction dims are {1} and {0}, which is matrix multiplication
                auto TC = graph.control.addElement(
                    ControlHypergraph::TensorContraction(A, B, {1}, {0}));
                m_op.insert_or_assign(mul.dest, TC);

                auto loadA = m_op.at(mul.a);
                auto loadB = m_op.at(mul.b);

                graph.control.addElement(ControlHypergraph::Sequence(), {loadA}, {TC});
                graph.control.addElement(ControlHypergraph::Sequence(), {loadB}, {TC});

                graph.mapper.connect<CoordGraph::MacroTile>(TC, D);

#ifndef NDEBUG
                auto parents = graph.control.parentNodes(TC).to<std::vector>();
                AssertFatal(parents.size() == 2, "T_MUL requires 2 inputs.");
                for(auto const& parent : parents)
                {
                    auto element = graph.control.getElement(parent);
                    auto node
                        = std::get<ControlHypergraph::Operation>(graph.control.getElement(parent));
                    AssertFatal(std::holds_alternative<ControlHypergraph::LoadTiled>(node),
                                "T_MUL inputs must be tiled.");
                }
#endif
            }

            /*
             * Nops... don't do anything
             */
            void operator()(Operations::Nop const& x) {}

            /*
             * Dispatch!
             */
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

        // Rename this when graph rearch complete
        KernelHypergraph translate2(std::shared_ptr<Command> command)
        {
            TIMER(t, "KernelGraph::translate");
            rocRoller::Log::getLogger()->debug("KernelGraph::translate(); Command\n{}",
                                               command->toString());
            TranslateVisitor2 visitor;
            return visitor(command);
        }

    }
}
