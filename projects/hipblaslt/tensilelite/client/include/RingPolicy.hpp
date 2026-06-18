// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        // HIP-free enablement policy for the warm ring. RingSlotController stays
        // pure accounting; this layer decides whether the client may use it.
        struct RingPolicyInputs
        {
            // Raw benchmark-loop count carried through for policy completeness.
            int    numBenchmarks         = 0;
            int    numEnqueuesPerSync    = 0;
            int    maxEnqueuesPerSync    = -1;
            int    numSyncsPerBenchmark  = 0;
            size_t minFlopsPerSync       = 0;
            int    numElementsToValidate = 0;
            // Mirrors ReferenceValidator::m_printAny; print-valids alone is not a driver.
            bool   printAny              = false;
        };

        struct RingPolicy
        {
            bool   allowed           = false;
            size_t activeBufferCount = 1;

            bool allocatesAltBuffers() const noexcept
            {
                return allowed && activeBufferCount > 1;
            }
        };

        inline bool hasValidationDriver(RingPolicyInputs const& inputs) noexcept
        {
            return inputs.numElementsToValidate != 0 || inputs.printAny;
        }

        inline bool benchmarkTimerRequestsSolutionRuns(RingPolicyInputs const& inputs) noexcept
        {
            // Raw solution-run request before max-enqueue capping.
            return inputs.numEnqueuesPerSync > 0 && inputs.numSyncsPerBenchmark > 0;
        }

        inline bool effectiveEnqueuesMayBePositive(RingPolicyInputs const& inputs) noexcept
        {
            // Conservative check for any positive enqueue count after min-flops and cap logic.
            if(inputs.maxEnqueuesPerSync == 0)
                return false;

            return inputs.numEnqueuesPerSync > 0 || inputs.minFlopsPerSync > 0;
        }

        inline bool benchmarkEnqueuesMayExecuteIfSolutionRuns(RingPolicyInputs const& inputs) noexcept
        {
            return inputs.numSyncsPerBenchmark > 0 && effectiveEnqueuesMayBePositive(inputs);
        }

        inline RingPolicy chooseRingPolicy(RingPolicyInputs const& inputs) noexcept
        {
            RingPolicy policy;

            // Only validation-driven runs with no possible benchmark enqueues may use the ring.
            if(!hasValidationDriver(inputs))
                return policy;

            if(benchmarkTimerRequestsSolutionRuns(inputs))
                return policy;

            if(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs))
                return policy;

            policy.allowed           = true;
            policy.activeBufferCount = 3;
            return policy;
        }
    } // namespace Client
} // namespace TensileLite
