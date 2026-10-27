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

// The backend interface of hipTensor. The solution layer (the public API, validation, plans
// and the plan cache) reaches kernels only through these types, so it carries no type of a
// kernel library. A backend (the Composable Kernel backend, the rocKE backend) decides at plan
// creation whether it serves a problem and prepares it; executing the plan only launches.

#ifndef HIPTENSOR_BACKEND_BACKEND_HPP
#define HIPTENSOR_BACKEND_BACKEND_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hiptensor/hiptensor_types.h>

#include "launch_config.hpp"
#include "problem.hpp"

namespace hiptensor
{
    namespace backend
    {
        // Backend names, as used by HIPTENSOR_BACKEND, logs and the plan cache.
        inline constexpr char kCkBackendName[]    = "ck";
        inline constexpr char kRockeBackendName[] = "rocke";

        // Identity of the kernel, or composite of kernels, a plan runs. `specKey` is the
        // backend's name for it; `uid` is FNV-1a of (backend, specKey, kernelAbi), a handle for
        // logs and plan-cache lines.
        struct KernelId
        {
            std::string backend;
            std::string specKey;
            uint32_t    kernelAbi = 0;
            uint64_t    uid       = 0;
        };

        uint64_t kernelUid(std::string_view backend, std::string_view specKey, uint32_t kernelAbi);

        // A capability answer. The reason is logged when a backend declines.
        struct Verdict
        {
            bool        admits = false;
            std::string reason;
        };

        // Host scalars of an execution, as the caller passed them: they are read with the
        // scalar type of the operation. Unused scalars are nullptr.
        struct Scalars
        {
            void const* alpha = nullptr;
            void const* beta  = nullptr;
            void const* gamma = nullptr;
        };

        // Device buffers of an execution. Inputs follow the tensor slots of the problem
        // (A, B, C, D); `out` is the output tensor (Problem::outputSlot()).
        struct Buffers
        {
            void const* A   = nullptr;
            void const* B   = nullptr;
            void const* C   = nullptr;
            void const* D   = nullptr;
            void*       out = nullptr;
        };

        // The library objects of a plan creation or an execution. Backends that keep their
        // state in the plan's preference (the Composable Kernel backend) use them; the others
        // only read the handle's device.
        struct PlanContext
        {
            hiptensorHandle_t handle             = nullptr;
            hiptensorPlan_t   plan               = nullptr;
            uint64_t          workspaceSizeLimit = 0;
        };

        // Workspace estimate request (hiptensorEstimateWorkspaceSize).
        struct EstimateContext
        {
            hiptensorHandle_t                   handle        = nullptr;
            hiptensorOperationDescriptor const* desc          = nullptr;
            hiptensorPlanPreference_t           planPref      = nullptr;
            hiptensorWorksizePreference_t       workspacePref = HIPTENSOR_WORKSPACE_DEFAULT;
        };

        // The prepared form of a plan: owned by the plan, created at plan creation and only
        // launched at execution.
        class IPrepared
        {
        public:
            virtual ~IPrepared() = default;

            virtual KernelId const& kernelId() const = 0;

            // Workspace the executions need, in bytes.
            virtual uint64_t workspaceSize() const = 0;

            // Launches the plan on cfg.stream.
            virtual hiptensorStatus_t run(PlanContext const&  context,
                                          Scalars const&      scalars,
                                          Buffers const&      buffers,
                                          void*               workspace,
                                          uint64_t            workspaceSize,
                                          LaunchConfig const& cfg) const
                = 0;
        };

        // A candidate strategy of a backend: one kernel or a composite of kernels.
        class ISolution
        {
        public:
            virtual ~ISolution() = default;

            virtual KernelId const& id() const = 0;

            // Host only: no allocation, no device call.
            virtual Verdict admits(Problem const& problem) const = 0;

            virtual uint64_t workspaceSize(Problem const& problem) const = 0;

            virtual hiptensorStatus_t prepare(Problem const&              problem,
                                              PlanContext const&          context,
                                              std::unique_ptr<IPrepared>& out) const
                = 0;
        };

        class IBackend
        {
        public:
            virtual ~IBackend() = default;

            virtual std::string_view name() const = 0;

            // Version of the kernel arguments and epilogue semantics of the backend's kernels.
            // Plan-cache files record it.
            virtual uint32_t kernelAbi() const = 0;

            // Whether the backend has kernels for the device. Never exits.
            virtual Verdict supportsArch(DeviceKey const& device) const = 0;

            // Candidates for the problem, best first after rank(). The default prepare() walks
            // them.
            virtual void enumerate(Problem const& problem, std::vector<ISolution const*>& out) const
            {
            }

            virtual void rank(Problem const& problem, std::vector<ISolution const*>& inOut) const {}

            // Prepares the problem at plan creation: HIPTENSOR_STATUS_NOT_SUPPORTED or
            // HIPTENSOR_STATUS_ARCH_MISMATCH with `reason` set when the backend declines it,
            // which lets another backend serve it. The default selects the first ranked
            // candidate that admits the problem, fits context.workspaceSizeLimit and prepares.
            virtual hiptensorStatus_t prepare(Problem const&              problem,
                                              PlanContext const&          context,
                                              std::unique_ptr<IPrepared>& out,
                                              std::string&                reason) const;

            // hiptensorEstimateWorkspaceSize for a problem this backend would serve.
            virtual hiptensorStatus_t estimateWorkspace(Problem const&         problem,
                                                        EstimateContext const& context,
                                                        uint64_t&              estimate,
                                                        std::string&           reason) const;

            // Persistent kernel cache (hiptensorWriteKernelCacheToFile /
            // hiptensorReadKernelCacheFromFile) of JIT-compiled kernels: the entries of the
            // device's class. A backend without one reports false.
            virtual bool hasKernelCache() const
            {
                return false;
            }

            virtual hiptensorStatus_t writeKernelCache(DeviceKey const& device,
                                                       char const*      fileName) const
            {
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }

            virtual hiptensorStatus_t readKernelCache(DeviceKey const& device, char const* fileName)
            {
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }
        };

        // Device memory owned by a plan (the A/B shadow output).
        struct DeviceAllocation
        {
            DeviceAllocation()                                   = default;
            DeviceAllocation(DeviceAllocation const&)            = delete;
            DeviceAllocation& operator=(DeviceAllocation const&) = delete;
            ~DeviceAllocation()
            {
                if(ptr != nullptr)
                {
                    static_cast<void>(hipFree(ptr));
                }
            }

            void*    ptr   = nullptr;
            uint64_t bytes = 0;
        };

        // What a plan holds of its backends: the prepared program of the backend that serves
        // it, and with HIPTENSOR_BACKEND_AB the one of the shadow backend whose result is
        // compared with it.
        struct PlanBackendState
        {
            Problem                    problem;
            std::unique_ptr<IPrepared> primary;
            std::unique_ptr<IPrepared> shadow;
            DeviceAllocation           shadowOutput;
        };
    } // namespace backend
} // namespace hiptensor

#endif // HIPTENSOR_BACKEND_BACKEND_HPP
