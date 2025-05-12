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

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Costs/Cost.hpp>
#include <rocRoller/Scheduling/Costs/LinearWeightedCost_fwd.hpp>

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

            /// Does our model predict that the instruction will be able to
            /// issue immediately, or after a stall?
            /// One per predicted stall cycle.
            float stallCycles;

            /// Penalizes instructions that are not MFMA, therefore prioritizing
            /// MFMA instructions.
            float notMFMA;

            /// Penalizes instructions that are MFMA, therefore prioritizing
            /// other instructions.
            float isMFMA;

            float isSMEM;
            float isSControl;
            float isSALU;

            float isVMEMRead;
            float isVMEMWrite;
            float isLDSRead;
            float isLDSWrite;
            float isVALU;

            float isACCVGPRWrite;
            float isACCVGPRRead;

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

            int vmemCycles    = 75;
            int vmemQueueSize = 4;

            int dsmemCycles    = 20;
            int dsmemQueueSize = 4;
        };

        /**
         * LinearWeightedCost: Orders the instructions based on a linear combination of a number of factors.
         */
        class LinearWeightedCost : public Cost
        {
        public:
            LinearWeightedCost(ContextPtr);

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

            Weights const& getWeights() const;

        private:
            Weights loadWeights(ContextPtr context) const;

            Weights m_weights;
        };
    }
}
