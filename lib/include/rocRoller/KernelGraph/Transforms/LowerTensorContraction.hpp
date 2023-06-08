#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Lower a TensorContraction operation.
         *
         * Currently supports matrix-matrix products.  The contraction
         * is lowered using a "data-parallel" decomposition.
         */
        class LowerTensorContraction : public GraphTransform
        {
        public:
            LowerTensorContraction(std::shared_ptr<CommandParameters> params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LowerTensorContraction";
            }

        private:
            std::shared_ptr<CommandParameters> m_params;
            ContextPtr                         m_context;
        };
    }
}
