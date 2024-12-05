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

            std::optional<int>
                createTailLoop(KernelGraph& graph, int tag, int unrollAmount, int unrollDimension);

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
