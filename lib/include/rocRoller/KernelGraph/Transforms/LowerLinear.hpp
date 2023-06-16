#pragma once
#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Expression_fwd.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Rewrite KernelGraph to distribute linear packets
         * onto GPU.
         *
         * Linear dimensions are packed/flattened, tiled onto
         * workgroups and wavefronts, and then operated on.
         */
        class LowerLinear : public GraphTransform
        {
        public:
            LowerLinear(ContextPtr context)
                : m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LowerLinear";
            }

        private:
            ContextPtr m_context;
        };

        /**
         * Rewrite KernelGraph to additionally distribute linear dimensions onto a For loop.
         */
        class LowerLinearLoop : public GraphTransform
        {
        public:
            LowerLinearLoop(Expression::ExpressionPtr loopSize, ContextPtr context)
                : m_loopSize(loopSize)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LowerLinearLoop";
            }

        private:
            ContextPtr                m_context;
            Expression::ExpressionPtr m_loopSize;
        };
    }
}
