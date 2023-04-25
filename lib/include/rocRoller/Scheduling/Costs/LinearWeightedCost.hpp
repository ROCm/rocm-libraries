
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Cost.hpp"
#include "LinearWeightedCost_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * Configures the relative weights used to penalize certain instructions.
         *
         * For `float` members, the comments generally describe what value is
         * multiplied by each coefficient to arrive at the cost of a particular
         * function.
         */
        struct Weights
        {
            Weights();
            /// Does the instruction require a nop before it?
            /// One per nop.
            float nops;

            /// Does the instruction require a `s_waitcnt vmcnt()` before it?
            /// One per value less than the max vmcnt for the architecture.
            float vmcnt;

            /// Does the instruction require a `s_waitcnt lgkmcnt()` before it?
            /// One per value less than the max lgkmcnt for the architecture.
            float lgkmcnt;

            /// `vectorQueueSat` penalizes an instruction for how much it will
            /// make the vector memory queue longer than `vmQueueLen`.
            int   vmQueueLen;
            float vectorQueueSat;

            /// `ldsQueueSat` penalizes an instruction for how much it will
            /// make the vector memory queue longer than `vmQueueLen`.
            float ldsQueueSat;
            int   lgkmQueueLen;

            /// Penalizes instructions that are not MFMA, therefore prioritizing
            /// MFMA instructions.
            float notMFMA;

            /// How many new SGPRs will the instruction allocate?
            float newSGPRs;

            /// How many new VGPRs will the instruction allocate?
            float newVGPRs;

            /// By how much will the instruction increase the high water mark
            /// of SGPR allocation so far seen in the kernel?
            float highWaterMarkSGPRs;

            /// By how much will the instruction increase the high water mark
            /// of VGPR allocation so far seen in the kernel?
            float highWaterMarkVGPRs;

            /// What fraction of the remaining SGPRs will be newly allocated by
            /// this instruction?
            float fractionOfSGPRs;

            /// What fraction of the remaining VGPRs will be newly allocated by
            /// this instruction?
            float fractionOfVGPRs;

            /// Will this instruction cause the kernel to run out of registers
            /// if scheduled right now?
            /// This should be set relatively high in order to cause us to pick
            /// any other instruction which will (hopefully) eventually free some
            /// registers.
            float outOfRegisters;

            /// If `true`, an `s_barrier` instruction that does not require a
            /// `s_waitcnt` will be given a cost of 0.  This will attempt to
            /// place redundant `s_barrier` instructions near to each other to
            /// hopefully reduce their impact on performance.
            /// TODO: Remove once we have removed the redundant barrier nodes
            /// from the control graph.
            bool zeroFreeBarriers;
        };

        /**
         * LinearWeightedCost: Orders the instructions based on a linear combination of a number of factors.
         */
        class LinearWeightedCost : public Cost
        {
        public:
            LinearWeightedCost(std::shared_ptr<Context>);

            using Base = Cost;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `CostFunction` is LinearWeighted
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `LinearWeightedCost` built from context
             */
            static std::shared_ptr<Cost> Build(Argument arg);

            /**
             * Return Name of `LinearWeightedCost`, used for debugging purposes currently
             */
            std::string name() const override;

            /**
             * Call operator orders the instructions.
             */
            float cost(Instruction const& inst, InstructionStatus const& status) const override;

        private:
            Weights m_weights;
        };
    }
}
