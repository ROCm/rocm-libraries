#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Performs the loop cleaning transformation
         *
         * Removes forloops that only contain a single iterations
         *
         * @return KernelGraph
         */
        class ConnectWorkgroups : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "ConnectWorkgroups";
            }
        };
    }
}
