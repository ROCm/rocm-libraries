#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the Loop Fusion transformation.
         *
         * Fuses multiple loops together if they iterate over the same
         * length.
         */
        class FuseLoops : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "FuseLoops";
            }
        };
    }
}
