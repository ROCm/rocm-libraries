/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
