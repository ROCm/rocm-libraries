
#pragma once

#include <memory>

#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    namespace Client
    {
        /**
         * Allows a coordinate graph from a kernel to be interrogated for the relationship
         * between kernel dimensions and memory dimensions by replacing sizes and strides with
         * literal values, and supplying default literal values to common dimensions in a
         * Transformer. See visualize.cpp for example usage.
         */
        class GraphInspector
        {
        public:
            GraphInspector(std::shared_ptr<Command> command,
                           CommandKernel&           kernel,
                           KernelArguments const&   runtimeArgs);

            /**
             * In the transformer, sets the value for all dimensions of type `DimensionType` to `val`.
             */
            template <KernelGraph::CoordinateGraph::CDimension DimensionType>
            void setCoordinate(CommandArgumentValue val);

            /**
             * In the transformer, sets the value for all dimensions matching `predicate` to `val`.
             */
            template <std::predicate<int> Predicate>
            void setCoordinate(Predicate predicate, CommandArgumentValue val);

            /**
             * In the transformer, sets the value for the single dimension matching `idx` to `val`.
             */
            void setCoordinate(int idx, CommandArgumentValue val);

            /**
             * In the transformer, sets the value for all dimensions in `indices` to `val`.
             */
            template <rocRoller::CForwardRangeOf<int> Range>
            void setCoordinate(Range const& indices, CommandArgumentValue val);

            /**
             * Returns the node ID for the LoadTiled operation with the given user tag.  For example,
             * sending in 1 should return the User dimension associated with the B matrix.
             */
            int findLoadTile(int userTag);

            /**
             * Returns the node ID for the StoreTiled operation with the given user tag.  For example,
             * sending in 0 should return the User dimension associated with the D matrix.
             */
            int findStoreTile(int userTag);

            /**
             * Sets arbitrary, reasonable values for many of the common dimensions needed to get
             *
             * Currently, all values are hard coded, so they may not be within bounds for all kernels.
             */
            void inventExecutionCoordinates();

            /**
             * Determines and then evaluates the expression for a given load dimension in the coordinate
             * graph. All dependent dimensions must have values via a combination of
             * assignLiteralSizesAndStrides(), inventExecutionCoordinates(), and setCoordinate().
             */
            size_t getLoadIndex(int coord);

            /**
             * Determines and then evaluates the expression for a given store dimension in the coordinate
             * graph. All dependent dimensions must have values via a combination of
             * assignLiteralSizesAndStrides(), inventExecutionCoordinates(), and setCoordinate().
             */
            size_t getStoreIndex(int coord);

            /**
             * @brief Get the MacroTileSizes
             *
             * @return std::tuple<int, int, int> {M, N, K}
             */
            std::tuple<int, int, int> getMacroTileSizes() const;

            /**
             * Get launch bounds for the given arguments.
             */
            KernelInvocation const& kernelInvocation() const;

            std::map<std::string, CommandArgumentValue> const& argValues();

            std::shared_ptr<KernelGraph::CoordinateGraph::CoordinateGraph> coords();

            KernelGraph::CoordinateGraph::Transformer& tx();

        private:
            void assignLiteralSizesAndStrides();

            /**
             * Finds and returns the node ID for the load or store with a given command argument name.
             * E.g. "Load_Tiled_0_pointer" should give the User dimension associated with the A matrix.
             */
            int findLoadStoreTile(std::string const& argName);

            std::shared_ptr<Command> m_command;
            CommandKernel&           m_kernel;
            KernelArguments          m_runtimeArgs;

            KernelGraph::KernelGraph                                       m_kgraph;
            std::shared_ptr<KernelGraph::CoordinateGraph::CoordinateGraph> m_coords;

            KernelGraph::CoordinateGraph::Transformer m_tx;
            KernelInvocation                          m_invocation;

            std::map<std::string, CommandArgumentValue> m_argValues;
        };

        /**
         * Returns a predicate `pred(int elem) -> bool` which will return true if elem is a node of type
         * T in the graph `coords`.
         */
        template <typename T>
        auto
            isNode(std::shared_ptr<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph> coords)
        {
            return [coords](int elem) {
                auto const& el = coords->getElement(elem);
                return coords->getElementType(elem) == Graph::ElementType::Node
                       && std::holds_alternative<T>(
                           std::get<KernelGraph::CoordinateGraph::Dimension>(el));
            };
        }

        /**
         * T must be a subclass of SubDimension
         *
         * Returns a predicate `pred(int elem) -> bool` which will return true if elem refers to:
         *  - a node of type T in `coords`
         *  - the `dim` field of the node must be equal to `dim`.
         */
        template <typename T>
        auto isSubDim(
            std::shared_ptr<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph> coords,
            int                                                                       dim)
        {
            return [coords, dim](int elem) {
                auto const& el = coords->getElement(elem);
                if(coords->getElementType(elem) != Graph::ElementType::Node)
                    return false;

                auto const& dimEl = std::get<KernelGraph::CoordinateGraph::Dimension>(el);

                if(!std::holds_alternative<T>(dimEl))
                    return false;

                auto const& t = std::get<T>(dimEl);

                return t.dim == dim;
            };
        }

        /**
         * Visitor which will convert a CommandArgumentValue to `size_t`.
         */
        inline auto to_size_t = [](auto val) -> size_t {
            using T = std::decay_t<decltype(val)>;
            if constexpr(std::is_pointer_v<T>)
                Throw<FatalError>("Unexpected pointer!");
            else
                return static_cast<size_t>(val);
        };

    }
}

#include "GraphInspector_impl.hpp"
