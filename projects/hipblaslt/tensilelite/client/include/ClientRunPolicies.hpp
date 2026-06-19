// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        using KernelHotPathSizeFn = std::function<std::uintmax_t(std::string const&)>;

        struct RotatingOutputPlan
        {
            int32_t maxRotatingBufferNum = 0;
            size_t  warmupRuns           = 0;
            size_t  syncs                = 0;
            size_t  enqueuesPerSync      = 0;
        };

        class RotatingOutputPolicy
        {
        public:
            RotatingOutputPlan plan(size_t warmupRuns,
                                    size_t syncs,
                                    size_t enqueuesPerSync) const;
        };

        struct IcacheAutoRotationPlan
        {
            int           extrasFromDataInit  = 0;
            int           extrasFromCache     = 0;
            int           extras              = 0;
            std::uintmax_t kernelHotPathSize  = 0;
            std::uintmax_t cacheBudgetBytes   = 0;
        };

        class IcacheRotationCursor
        {
        public:
            int  nextIndex(int moduleCount);
            void reset();

        private:
            size_t m_launchIndex = 0;
        };

        class IcacheRotationPolicy
        {
        public:
            static KernelHotPathSizeFn defaultKernelHotPathSizeFn();

            bool shouldLoadAutoCopies(int requestedCopies, int currentModuleCount) const;

            IcacheAutoRotationPlan computeAutoPlan(
                size_t                              inputSlotCount,
                std::vector<std::string> const&     codeObjectFilenames,
                int                                 rotateSizeKB,
                KernelHotPathSizeFn const&          kernelHotPathSizeFn) const;
        };
    } // namespace Client
} // namespace TensileLite
