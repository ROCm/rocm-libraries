/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_BACKEND_REGISTRY_HPP
#define HIPTENSOR_BACKEND_REGISTRY_HPP

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "backend.hpp"

namespace hiptensor
{
    namespace backend
    {
        // The backends built into the library, in decreasing priority.
        class BackendRegistry
        {
        public:
            static BackendRegistry& instance();

            void add(std::unique_ptr<IBackend> backend, int priority);

            // nullptr if no backend of that name is built.
            IBackend* find(std::string_view name) const;

            std::vector<IBackend*> backends() const;

        private:
            BackendRegistry();

            struct Entry
            {
                std::unique_ptr<IBackend> backend;
                int                       priority;
            };

            mutable std::mutex mMutex;
            std::vector<Entry> mEntries;
        };

        // Backend selection from the environment, read at every plan creation and workspace
        // estimate:
        //   HIPTENSOR_BACKEND=auto|ck|rocke       the backend that serves plans (auto: the default
        //                                         backend, Composable Kernel)
        //   HIPTENSOR_BACKEND_FALLBACK=1|0        when the selected backend declines a problem,
        //                                         the Composable Kernel backend serves it (default 1)
        //   HIPTENSOR_BACKEND_AB=<primary>,<shadow>
        //                                         debug: plans prepare both backends, every
        //                                         execution runs the shadow backend into a buffer
        //                                         of its own first, then the primary backend into
        //                                         the caller's output, and logs how far the two
        //                                         results are apart (heuristics trace). Overrides
        //                                         HIPTENSOR_BACKEND.
        struct BackendPolicy
        {
            std::string primary;
            bool        fallback = true;
            std::string shadow;

            // HIPTENSOR_STATUS_INVALID_VALUE with `error` set for a value it does not know.
            static hiptensorStatus_t fromEnvironment(BackendPolicy& policy, std::string& error);

            // The backends to try, in order: the selected one, then the fallback.
            std::vector<std::string> order() const;
        };

        // Selects and prepares the backend of a new plan whose descriptors and preference are
        // already owned by the plan. Sets plan->mBackend and plan->mRequiredWorkspace.
        hiptensorStatus_t createPlanBackend(hiptensorHandle_t handle,
                                            hiptensorPlan_t   plan,
                                            uint64_t          workspaceSizeLimit);

        // Runs a plan through its backends (A/B shadow first, when there is one).
        hiptensorStatus_t executePlan(hiptensorHandle_t   handle,
                                      hiptensorPlan_t     plan,
                                      Scalars const&      scalars,
                                      Buffers const&      buffers,
                                      void*               workspace,
                                      uint64_t            workspaceSize,
                                      LaunchConfig const& cfg);

        // hiptensorEstimateWorkspaceSize through the backend a plan of desc would select.
        hiptensorStatus_t estimateWorkspace(EstimateContext const& context, uint64_t& estimate);

        // The kernel cache of the first backend that has one, for the handle's device;
        // HIPTENSOR_STATUS_SUCCESS without touching the file when no backend keeps one.
        hiptensorStatus_t writeKernelCache(hiptensorHandle_t handle, char const* fileName);
        hiptensorStatus_t readKernelCache(hiptensorHandle_t handle, char const* fileName);
    } // namespace backend
} // namespace hiptensor

#endif // HIPTENSOR_BACKEND_REGISTRY_HPP
