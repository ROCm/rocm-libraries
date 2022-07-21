#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using CoordinateTransform::MacroTile;

        namespace Expression = rocRoller::Expression;

        /***********************************
         * Command to HyperGraph translator
         */

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
                for(int i = 0; i < sizes.size(); ++i)
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
                                         {ControlGraph::LoadLinear(load.getTag(), user, linear)},
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
                                         {ControlGraph::LoadVGPR(load.getTag(), user)},
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
                for(int i = 0; i < sizes.size(); ++i)
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

                m_kgraph.control.addEdge({ControlGraph::Kernel()},
                                         {ControlGraph::LoadTiled(tag, user, tiled)},
                                         ControlGraph::Body());
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
                for(int i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim         = CoordinateTransform::SubDimension(
                        store.getTag(), i, nullptr, stride_expr, true);
                    dims.push_back(dim);
                }

                auto inLinear  = CoordinateTransform::Linear(store.getTag());
                auto outLinear = CoordinateTransform::Linear(store.getTag(), true);

                m_kgraph.coordinates.addEdge(
                    {inLinear}, {outLinear}, CoordinateTransform::MakeOutput());
                m_kgraph.coordinates.addEdge({outLinear}, dims, CoordinateTransform::Split());

                auto user = CoordinateTransform::User(store.getTag(), store.data()->name(), true);
                m_kgraph.coordinates.addEdge(dims, {user}, CoordinateTransform::Join());
                m_kgraph.coordinates.addEdge({inLinear}, {user}, CoordinateTransform::DataFlow());

                m_kgraph.control.addEdge(
                    {user}, {ControlGraph::StoreLinear(store.getTag(), inLinear, user)});
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
                for(int i = 0; i < strides.size(); ++i)
                {
                    auto stride_expr = std::make_shared<Expression::Expression>(strides[i]);
                    auto dim         = CoordinateTransform::SubDimension(
                        store.getTag(), i, nullptr, stride_expr, true);
                    dims.push_back(dim);
                }

                auto inTile  = CoordinateTransform::MacroTile(store.getTag(), dims.size());
                auto outTile = CoordinateTransform::MacroTile(store.getTag(), dims.size(), true);

                m_kgraph.coordinates.addEdge(
                    {inTile}, {outTile}, CoordinateTransform::MakeOutput());
                m_kgraph.coordinates.addEdge(
                    {outTile}, dims, CoordinateTransform::DestructMacroTile());

                auto user = CoordinateTransform::User(store.getTag(), store.data()->name(), true);
                m_kgraph.coordinates.addEdge(dims, {user}, CoordinateTransform::Join());

                m_kgraph.control.addEdge({user},
                                         {ControlGraph::StoreTiled(store.getTag(), outTile, user)});
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

                auto loadA = ControlGraph::LoadTiled(A.tag, CoordinateTransform::User(A.tag), A);
                auto loadB = ControlGraph::LoadTiled(B.tag, CoordinateTransform::User(B.tag), B);

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

        KernelGraph translate(std::shared_ptr<Command> command)
        {
            TIMER(t, "KernelGraph::translate");
            rocRoller::Log::getLogger()->debug("KernelGraph::translate(); Command\n{}",
                                               command->toString());
            TranslateVisitor visitor;
            return visitor(command);
        }
    }
}
