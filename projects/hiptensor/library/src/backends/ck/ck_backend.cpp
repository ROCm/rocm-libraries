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

#include "ck_backend.hpp"

#include "ck_tensor_limits.hpp"
#include "contraction/contraction_solution.hpp"
#include "data_types.hpp"
#include "hiptensor_options.hpp"

// Plan creation and workspace estimates of the CK solution code (hiptensor_contraction.cpp).
hiptensorStatus_t contractionInitPlan(const hiptensorHandle_t              handle,
                                      hiptensorPlan_t                      plan,
                                      const hiptensorOperationDescriptor_t desc,
                                      const hiptensorPlanPreference_t      pref,
                                      uint64_t                             workspaceSizeLimit);
hiptensorStatus_t contractionTrinaryInitPlan(const hiptensorHandle_t              handle,
                                             hiptensorPlan_t                      plan,
                                             const hiptensorOperationDescriptor_t desc,
                                             const hiptensorPlanPreference_t      pref,
                                             uint64_t workspaceSizeLimit);
hiptensorStatus_t contractionGetWorkspaceSize(const hiptensorHandle_t              handle,
                                              const hiptensorOperationDescriptor_t desc,
                                              const hiptensorPlanPreference_t      planPref,
                                              const hiptensorWorksizePreference_t  workspacePref,
                                              uint64_t* workspaceSizeEstimate);
hiptensorStatus_t
    contractionTrinaryGetWorkspaceSize(const hiptensorHandle_t              handle,
                                       const hiptensorOperationDescriptor_t desc,
                                       const hiptensorPlanPreference_t      planPref,
                                       const hiptensorWorksizePreference_t  workspacePref,
                                       uint64_t*                            workspaceSizeEstimate);

namespace hiptensor
{
    hiptensorStatus_t elementwiseCheckPlan(hiptensorOperationDescriptor const& desc);
    hiptensorStatus_t reductionCheckPlan(hiptensorOperationDescriptor const& desc);

    namespace ckbackend
    {
        namespace
        {
            using backend::Buffers;
            using backend::KernelId;
            using backend::LaunchConfig;
            using backend::OpKind;
            using backend::PlanContext;
            using backend::Problem;
            using backend::Scalars;

            // The CK program of a plan. The CK solution code keeps the selected kernel in the
            // plan's preference, so the prepared program only records the operation and runs
            // the CK execution of that operation.
            class CkPrepared final : public backend::IPrepared
            {
            public:
                CkPrepared(OpKind kind, KernelId id, uint64_t workspaceSize)
                    : mKind(kind)
                    , mId(std::move(id))
                    , mWorkspaceSize(workspaceSize)
                {
                }

                KernelId const& kernelId() const override
                {
                    return mId;
                }

                uint64_t workspaceSize() const override
                {
                    return mWorkspaceSize;
                }

                hiptensorStatus_t run(PlanContext const&  context,
                                      Scalars const&      scalars,
                                      Buffers const&      buffers,
                                      void*               workspace,
                                      uint64_t            workspaceSize,
                                      LaunchConfig const& cfg) const override
                {
                    auto handle = context.handle;
                    auto plan   = context.plan;
                    switch(mKind)
                    {
                    case OpKind::Contraction:
                        return contract(handle,
                                        plan,
                                        scalars.alpha,
                                        buffers.A,
                                        buffers.B,
                                        scalars.beta,
                                        buffers.C,
                                        buffers.out,
                                        workspace,
                                        workspaceSize,
                                        cfg);
                    case OpKind::ContractionTrinary:
                        return contractTrinary(handle,
                                               plan,
                                               scalars.alpha,
                                               buffers.A,
                                               buffers.B,
                                               buffers.C,
                                               scalars.beta,
                                               buffers.D,
                                               buffers.out,
                                               workspace,
                                               workspaceSize,
                                               cfg);
                    case OpKind::Permute:
                        return permute(handle, plan, scalars.alpha, buffers.A, buffers.out, cfg);
                    case OpKind::ElementwiseBinary:
                        return elementwiseBinary(handle,
                                                 plan,
                                                 scalars.alpha,
                                                 buffers.A,
                                                 scalars.gamma,
                                                 buffers.C,
                                                 buffers.out,
                                                 cfg);
                    case OpKind::ElementwiseTrinary:
                        return elementwiseTrinary(handle,
                                                  plan,
                                                  scalars.alpha,
                                                  buffers.A,
                                                  scalars.beta,
                                                  buffers.B,
                                                  scalars.gamma,
                                                  buffers.C,
                                                  buffers.out,
                                                  cfg);
                    case OpKind::Reduction:
                        return reduce(handle,
                                      plan,
                                      scalars.alpha,
                                      buffers.A,
                                      scalars.beta,
                                      buffers.C,
                                      buffers.out,
                                      workspace,
                                      workspaceSize,
                                      cfg);
                    }
                    return HIPTENSOR_STATUS_INTERNAL_ERROR;
                }

            private:
                OpKind   mKind;
                KernelId mId;
                uint64_t mWorkspaceSize;
            };

            class CkBackend final : public backend::IBackend
            {
            public:
                std::string_view name() const override
                {
                    return backend::kCkBackendName;
                }

                uint32_t kernelAbi() const override
                {
                    return kKernelAbi;
                }

                // The handle already refused the devices CK has no kernels for.
                backend::Verdict supportsArch(backend::DeviceKey const& device) const override
                {
                    return {true, ""};
                }

                hiptensorStatus_t prepare(Problem const&                       problem,
                                          PlanContext const&                   context,
                                          std::unique_ptr<backend::IPrepared>& out,
                                          std::string& reason) const override
                {
                    auto plan = context.plan;
                    auto desc = plan->mOpDesc;

                    // The kernels would silently compute wrong results for tensors they cannot
                    // address, so no plan is created for them.
                    if(auto status = checkCkTensorLimits(*desc, "hiptensorCreatePlan");
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        reason = "a tensor exceeds the limits of the CK kernels";
                        return status;
                    }

                    // Problems the elementwise and reduction kernels cannot run are reported
                    // here, rather than as HIPTENSOR_STATUS_INTERNAL_ERROR when the plan is
                    // executed.
                    auto status = HIPTENSOR_STATUS_SUCCESS;
                    switch(problem.kind)
                    {
                    case OpKind::Permute:
                    case OpKind::ElementwiseBinary:
                    case OpKind::ElementwiseTrinary:
                        status = elementwiseCheckPlan(*desc);
                        break;
                    case OpKind::Reduction:
                        status = reductionCheckPlan(*desc);
                        break;
                    case OpKind::Contraction:
                        status = contractionInitPlan(
                            context.handle, plan, desc, plan->mPref, context.workspaceSizeLimit);
                        break;
                    case OpKind::ContractionTrinary:
                        status = contractionTrinaryInitPlan(
                            context.handle, plan, desc, plan->mPref, context.workspaceSizeLimit);
                        break;
                    }
                    if(status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        reason = "the CK kernels do not support the problem";
                        return status;
                    }

                    // Contraction kernels are selected now; the elementwise and reduction
                    // kernels at the first execution.
                    KernelId id;
                    id.backend     = backend::kCkBackendName;
                    id.kernelAbi   = kKernelAbi;
                    auto* solution = static_cast<ContractionSolution*>(plan->mPref->mSolution);
                    if((problem.kind == OpKind::Contraction
                        || problem.kind == OpKind::ContractionTrinary)
                       && solution != nullptr)
                    {
                        id.specKey = solution->kernelName();
                        id.uid     = solution->uid();
                    }
                    else
                    {
                        id.specKey = "selected at the first execution";
                    }

                    out = std::make_unique<CkPrepared>(
                        problem.kind, std::move(id), plan->mRequiredWorkspace);
                    return HIPTENSOR_STATUS_SUCCESS;
                }

                hiptensorStatus_t estimateWorkspace(Problem const&                  problem,
                                                    backend::EstimateContext const& context,
                                                    uint64_t&                       estimate,
                                                    std::string& reason) const override
                {
                    auto desc = const_cast<hiptensorOperationDescriptor*>(context.desc);
                    switch(problem.kind)
                    {
                    case OpKind::Contraction:
                        return contractionGetWorkspaceSize(context.handle,
                                                           desc,
                                                           context.planPref,
                                                           context.workspacePref,
                                                           &estimate);
                    case OpKind::ContractionTrinary:
                        return contractionTrinaryGetWorkspaceSize(context.handle,
                                                                  desc,
                                                                  context.planPref,
                                                                  context.workspacePref,
                                                                  &estimate);
                    default:
                        // Elementwise operations and reductions need no workspace.
                        estimate = 0u;
                        return HIPTENSOR_STATUS_SUCCESS;
                    }
                }
            };
        } // namespace

        std::unique_ptr<backend::IBackend> makeCkBackend()
        {
            return std::make_unique<CkBackend>();
        }

        backend::LaunchConfig timedLaunch(backend::LaunchConfig const& cfg)
        {
            auto& options = HiptensorOptions::instance();

            auto timed       = cfg;
            timed.timeKernel = true;
            timed.coldIters  = options->coldRuns();
            timed.hotIters   = options->hotRuns();
            return timed;
        }
    } // namespace ckbackend
} // namespace hiptensor
