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
#include <rocRoller/CommandSolution_fwd.hpp>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {

        /**
         * @brief Gets the name of the given for loop.
         *
         * @param graph
         * @param start
         * @return std::string
         */
        std::string getForLoopName(KernelGraph& graph, int start);

        /**
         * @brief Determine how many times to unroll the loop.
         *
         * A value of 0 or 1 means do not unroll it.
         * Use getForLoopName to determine which forLoop we are attempting to unroll
         */
        unsigned int
            getUnrollAmount(KernelGraph& graph, int loopTag, CommandParametersPtr const& params);

        /**
         * @brief Performs the Loop Unrolling transformation.
         *
         * Unrolls every loop that does not have a previous iteration
         * dependency by a value of 2.
         */
        class UnrollLoops : public GraphTransform
        {
        public:
            UnrollLoops(CommandParametersPtr params, ContextPtr context);

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "UnrollLoops";
            }

            std::optional<int> createTailLoop(KernelGraph& graph,
                                              int          tag,
                                              int          unrollAmount,
                                              int          unrollDimension,
                                              int          forLoopDimension);

        private:
            int  createUnrollDimension(KernelGraph& graph, int forLoopDimension, int unrollAmount);
            void unrollLoop(KernelGraph& graph, int tag);
            void commit(KernelGraph& kgraph);

            std::map<int, int>                                             m_unrolledLoopDimensions;
            std::map<std::pair<int, int>, std::shared_ptr<GraphReindexer>> m_unrollReindexers;
            std::unordered_set<int>                                        m_unrolledLoopOps;
            ContextPtr                                                     m_context;
            CommandParametersPtr                                           m_params;
        };

    }
}
