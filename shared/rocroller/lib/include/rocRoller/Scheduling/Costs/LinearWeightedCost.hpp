// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            float nops = 0.0f;

            /// Does the instruction require a `s_waitcnt vmcnt()` before it?
            /// One per value less than the max vmcnt for the architecture.
            float vmcnt = 0.0f;

            /// Does the instruction require a `s_waitcnt lgkmcnt()` before it?
            /// One per value less than the max lgkmcnt for the architecture.
            float lgkmcnt = 0.0f;

            /// `vectorQueueSat` penalizes an instruction for how much it will
            /// make the vector memory queue longer than `vmQueueLen`.
            int   vmQueueLen     = 0;
            float vectorQueueSat = 0.0f;

            /// `ldsQueueSat` penalizes an instruction for how much it will
            /// make the vector memory queue longer than `vmQueueLen`.
            float ldsQueueSat  = 0.0f;
            int   lgkmQueueLen = 0;

            /// Does our model predict that the instruction will be able to
            /// issue immediately, or after a stall?
            /// One per predicted stall cycle.
            float stallCycles = 0.0f;

            /// Penalizes instructions that are not MFMA, therefore prioritizing
            /// MFMA instructions.
            float notMFMA = 0.0f;

            /// Penalizes instructions that are MFMA, therefore prioritizing
            /// other instructions.
            float isMFMA = 0.0f;

            float isSMEM     = 0.0f;
            float isSControl = 0.0f;
            float isSALU     = 0.0f;

            float isVMEMRead  = 0.0f;
            float isVMEMWrite = 0.0f;
            float isLDSRead   = 0.0f;
            float isLDSWrite  = 0.0f;
            float isVALU      = 0.0f;

            float isACCVGPRWrite = 0.0f;
            float isACCVGPRRead  = 0.0f;

            /// How many new SGPRs will the instruction allocate?
            float newSGPRs = 0.0f;

            /// How many new VGPRs will the instruction allocate?
            float newVGPRs = 0.0f;

            /// By how much will the instruction increase the high water mark
            /// of SGPR allocation so far seen in the kernel?
            float highWaterMarkSGPRs = 0.0f;

            /// By how much will the instruction increase the high water mark
            /// of VGPR allocation so far seen in the kernel?
            float highWaterMarkVGPRs = 0.0f;

            /// What fraction of the remaining SGPRs will be newly allocated by
            /// this instruction?
            float fractionOfSGPRs = 0.0f;

            /// What fraction of the remaining VGPRs will be newly allocated by
            /// this instruction?
            float fractionOfVGPRs = 0.0f;

            /// Will this instruction cause the kernel to run out of registers
            /// if scheduled right now?
            /// This should be set relatively high in order to cause us to pick
            /// any other instruction which will (hopefully) eventually free some
            /// registers.
            float outOfRegisters = 0.0f;

            /// If `true`, an `s_barrier` instruction that does not require a
            /// `s_waitcnt` will be given a cost of 0.  This will attempt to
            /// place redundant `s_barrier` instructions near to each other to
            /// hopefully reduce their impact on performance.
            /// TODO: Remove once we have removed the redundant barrier nodes
            /// from the control graph.
            bool zeroFreeBarriers = false;

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
            LinearWeightedCost(ContextPtr ctx, CostFunction fn);

            using Base = Cost;

            static const std::string        Basename;
            inline static const std::string Name = "LinearWeightedCost";

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
            Weights loadWeights(ContextPtr ctx, CostFunction fn) const;

            Weights m_weights;
        };
    }
}
