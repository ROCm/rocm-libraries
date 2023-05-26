#pragma once
#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Base class for graph transformations.
         * Contains an apply function, that takes in a KernelGraph and returns a 
         * transformed kernel graph based on the transformation.
        */
        class GraphTransform
        {
        public:
            GraphTransform()                                       = default;
            ~GraphTransform()                                      = default;
            virtual KernelGraph apply(KernelGraph const& original) = 0;
            virtual std::string name() const                       = 0;
        };
    }
}
