#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the Sequentialize transformation.
         *
         * Sequentializes the epilogue components post forLoop.
         */
        class Sequentialize : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "Sequentialize";
            }
        };
    }
}
