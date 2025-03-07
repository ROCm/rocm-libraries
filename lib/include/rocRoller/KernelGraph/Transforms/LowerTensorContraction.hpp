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
            LowerTensorContraction(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "LowerTensorContraction";
            }

            std::vector<GraphConstraint> postConstraints() const override;

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
