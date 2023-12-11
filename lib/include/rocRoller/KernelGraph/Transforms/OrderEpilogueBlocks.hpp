#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the Order Epilogue Blocks transformation.
         *
         * Orders the epilogue components.
         */
        class OrderEpilogueBlocks : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderEpilogueBlocks";
            }
        };
    }
}
