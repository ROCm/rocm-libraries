#pragma once

#include <memory>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Base class for graph transformations.
         *
         * Contains an apply function, that takes in a KernelGraph and
         * returns a transformed kernel graph based on the
         * transformation.
         */
        class GraphTransform
        {
        public:
            GraphTransform()                                       = default;
            ~GraphTransform()                                      = default;
            virtual KernelGraph apply(KernelGraph const& original) = 0;
            virtual std::string name() const                       = 0;

            /**
             * @brief The list of assumptions that must hold before applying this transformation.
             */
            virtual std::vector<GraphConstraint> preConstraints() const
            {
                return {};
            }

            /**
             * @brief The list of ongoing assumptions that can be made after applying this transformation.
             */
            virtual std::vector<GraphConstraint> postConstraints() const
            {
                return {};
            }
        };
    }
}
