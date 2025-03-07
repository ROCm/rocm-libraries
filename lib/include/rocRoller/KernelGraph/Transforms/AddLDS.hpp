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
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Only SetCoordinate, LoadTiled, and Multiply nodes and Body and Sequence edges are allowed in the ForLoop.
         *
         * If there are ever any other nodes/edges, they will need to be handled.
         */
        ConstraintStatus AcceptablePrefetchNodes(const KernelGraph& k);

        /**
         * @brief Rewrite KernelGraph to add LDS operations for
         * loading/storing data.
         *
         * Modifies the coordinate and control graphs to add LDS
         * information.
         *
         * @ingroup Transformations
         */
        class AddLDS : public GraphTransform
        {
        public:
            AddLDS(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddLDS";
            }

            std::vector<GraphConstraint> postConstraints() const override;

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };

        /**
         * @brief Rewrite KernelGraph to add LDS operations for
         * loading/storing data.
         *
         * Modifies the coordinate and control graphs to add LDS
         * information.
         */
        class AddPrefetch : public GraphTransform
        {
        public:
            AddPrefetch(CommandParametersPtr params, ContextPtr context)
                : m_params(params)
                , m_context(context)
            {
            }

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "AddPrefetch";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {&AcceptablePrefetchNodes};
            }

        private:
            CommandParametersPtr m_params;
            ContextPtr           m_context;
        };
    }
}
