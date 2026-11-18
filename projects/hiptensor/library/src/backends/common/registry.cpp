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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include "backends/ck/ck_backend.hpp"
#ifdef HIPTENSOR_ENABLE_ROCKE_BACKEND
#include "backends/rocke/rocke_backend.hpp"
#endif
#include "data_types.hpp"
#include "logger.hpp"
#include "platform.hpp"
#include "registry.hpp"

namespace hiptensor
{
    namespace backend
    {
        namespace
        {
            // Registry priorities: only order backends that the selection does not name.
            constexpr int kCkPriority    = 10;
            constexpr int kRockePriority = 20;

            // The backend that serves plans unless HIPTENSOR_BACKEND names another one.
            constexpr char const* kDefaultBackend = kCkBackendName;

            // Backend names the environment may use, built or not.
            constexpr char const* kKnownBackends[] = {kCkBackendName, kRockeBackendName};

            bool isKnownBackend(std::string const& name)
            {
                return std::any_of(std::begin(kKnownBackends),
                                   std::end(kKnownBackends),
                                   [&name](char const* known) { return name == known; });
            }

            std::string lowerTrimmed(std::string value)
            {
                auto notSpace = [](unsigned char c) { return !std::isspace(c); };
                value.erase(value.begin(), std::find_if(value.begin(), value.end(), notSpace));
                value.erase(std::find_if(value.rbegin(), value.rend(), notSpace).base(),
                            value.end());
                std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                return value;
            }

            bool isDecline(hiptensorStatus_t status)
            {
                return status == HIPTENSOR_STATUS_NOT_SUPPORTED
                       || status == HIPTENSOR_STATUS_ARCH_MISMATCH;
            }

            // Prepares `problem` on the first backend of `order` that serves it. A backend that
            // is not built, or that declines the problem, passes it on; any other failure ends
            // the search. Returns the status of the last backend tried.
            hiptensorStatus_t prepareFirst(std::vector<std::string> const& order,
                                           Problem const&                  problem,
                                           PlanContext const&              context,
                                           std::unique_ptr<IPrepared>&     out,
                                           std::string&                    servedBy)
            {
                auto& logger = Logger::instance();
                char  msg[512];

                auto status = HIPTENSOR_STATUS_NOT_SUPPORTED;
                for(std::size_t i = 0; i < order.size(); i++)
                {
                    auto const& name    = order[i];
                    auto*       backend = BackendRegistry::instance().find(name);
                    std::string reason;
                    if(backend == nullptr)
                    {
                        status = HIPTENSOR_STATUS_NOT_SUPPORTED;
                        reason = "the backend is not built into this library";
                    }
                    else
                    {
                        out.reset();
                        status = backend->prepare(problem, context, out, reason);
                        if(status == HIPTENSOR_STATUS_SUCCESS && out == nullptr)
                        {
                            status = HIPTENSOR_STATUS_INTERNAL_ERROR;
                        }
                    }

                    if(status == HIPTENSOR_STATUS_SUCCESS)
                    {
                        servedBy = name;
                        return status;
                    }
                    if(!isDecline(status))
                    {
                        return status;
                    }

                    snprintf(msg,
                             sizeof(msg),
                             "Backend %s declined the %s problem: %s (%s)%s",
                             name.c_str(),
                             opKindName(problem.kind),
                             reason.c_str(),
                             hiptensorGetErrorString(status),
                             i + 1 < order.size() ? ", trying the next backend" : "");
                    logger->logHeuristics("hiptensorCreatePlan", msg);
                }
                return status;
            }

            // Reads element `offset` of a buffer of `type` as up to two doubles (real, imag).
            bool readElement(std::vector<uint8_t> const& bytes,
                             hiptensorDataType_t         type,
                             int64_t                     offset,
                             double&                     re,
                             double&                     im)
            {
                im                = 0.0;
                auto const* data  = bytes.data();
                auto const  index = static_cast<std::size_t>(offset);
                switch(type)
                {
                case HIPTENSOR_R_16F:
                {
                    _Float16 value;
                    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
                    re = static_cast<double>(value);
                    return true;
                }
                case HIPTENSOR_R_16BF:
                {
                    hip_bfloat16 value;
                    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
                    re = static_cast<double>(static_cast<float>(value));
                    return true;
                }
                case HIPTENSOR_R_32F:
                {
                    float value;
                    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
                    re = value;
                    return true;
                }
                case HIPTENSOR_R_64F:
                {
                    double value;
                    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
                    re = value;
                    return true;
                }
                case HIPTENSOR_C_32F:
                {
                    float value[2];
                    std::memcpy(value, data + index * sizeof(value), sizeof(value));
                    re = value[0];
                    im = value[1];
                    return true;
                }
                case HIPTENSOR_C_64F:
                {
                    double value[2];
                    std::memcpy(value, data + index * sizeof(value), sizeof(value));
                    re = value[0];
                    im = value[1];
                    return true;
                }
                default:
                    return false;
                }
            }

            // Compares the primary backend's output with the shadow backend's, over the elements
            // the output tensor addresses, and logs the difference.
            void compareOutputs(PlanBackendState const& state,
                                void const*             primaryOutput,
                                hipStream_t             stream)
            {
                auto& logger = Logger::instance();
                char  msg[512];

                auto const& output = state.problem.output();
                auto const  bytes  = state.shadowOutput.bytes;
                if(hipStreamSynchronize(stream) != hipSuccess)
                {
                    logger->logHeuristics("hiptensorExecute", "A/B: cannot synchronize the stream");
                    return;
                }

                std::vector<uint8_t> primary(bytes), shadow(bytes);
                if(hipMemcpy(primary.data(), primaryOutput, bytes, hipMemcpyDeviceToHost)
                       != hipSuccess
                   || hipMemcpy(shadow.data(), state.shadowOutput.ptr, bytes, hipMemcpyDeviceToHost)
                          != hipSuccess)
                {
                    logger->logHeuristics("hiptensorExecute", "A/B: cannot read the outputs");
                    return;
                }

                // Walk every index of the output tensor.
                auto const           rank = output.extents.size();
                std::vector<int64_t> index(rank, 0);
                double               maxAbsDiff = 0.0, sumSqDiff = 0.0, sumSqRef = 0.0;
                int64_t              count = 0, nonFinite = 0;
                while(true)
                {
                    int64_t offset = 0;
                    for(std::size_t i = 0; i < rank; i++)
                    {
                        offset += index[i] * output.strides[i];
                    }

                    double pr, pi, sr, si;
                    if(!readElement(primary, output.type, offset, pr, pi)
                       || !readElement(shadow, output.type, offset, sr, si))
                    {
                        logger->logHeuristics("hiptensorExecute",
                                              "A/B: the output data type is not compared");
                        return;
                    }
                    double const diff = std::hypot(pr - sr, pi - si);
                    if(!std::isfinite(diff))
                    {
                        // Equal infinities and NaN in both outputs agree.
                        bool const same
                            = (std::isnan(pr) && std::isnan(sr)) || (pr == sr && pi == si);
                        nonFinite += same ? 0 : 1;
                    }
                    else
                    {
                        maxAbsDiff = std::max(maxAbsDiff, diff);
                        sumSqDiff += diff * diff;
                        sumSqRef += sr * sr + si * si;
                    }
                    count++;

                    std::size_t mode = 0;
                    while(mode < rank && ++index[mode] == output.extents[mode])
                    {
                        index[mode] = 0;
                        mode++;
                    }
                    if(mode == rank)
                    {
                        break;
                    }
                }

                double const relFro
                    = sumSqRef > 0.0 ? std::sqrt(sumSqDiff / sumSqRef) : std::sqrt(sumSqDiff);
                snprintf(msg,
                         sizeof(msg),
                         "A/B %s vs %s: %lld elements, max |diff| %.6g, relative Frobenius "
                         "difference %.6g, non-finite mismatches %lld",
                         state.primary->kernelId().backend.c_str(),
                         state.shadow->kernelId().backend.c_str(),
                         static_cast<long long>(count),
                         maxAbsDiff,
                         relFro,
                         static_cast<long long>(nonFinite));
                logger->logHeuristics("hiptensorExecute", msg);
            }
        } // namespace

        uint64_t kernelUid(std::string_view backend, std::string_view specKey, uint32_t kernelAbi)
        {
            uint64_t hash = 14695981039346656037ull;
            auto     mix = [&hash](unsigned char byte) { hash = (hash ^ byte) * 1099511628211ull; };
            for(auto c : backend)
            {
                mix(static_cast<unsigned char>(c));
            }
            mix(0u);
            for(auto c : specKey)
            {
                mix(static_cast<unsigned char>(c));
            }
            mix(0u);
            for(int i = 0; i < 4; i++)
            {
                mix(static_cast<unsigned char>((kernelAbi >> (8 * i)) & 0xFFu));
            }
            return hash;
        }

        hiptensorStatus_t IBackend::prepare(Problem const&              problem,
                                            PlanContext const&          context,
                                            std::unique_ptr<IPrepared>& out,
                                            std::string&                reason) const
        {
            auto arch = supportsArch(problem.device);
            if(!arch.admits)
            {
                reason = arch.reason;
                return HIPTENSOR_STATUS_ARCH_MISMATCH;
            }

            std::vector<ISolution const*> candidates;
            enumerate(problem, candidates);
            rank(problem, candidates);

            reason = candidates.empty() ? "no kernel for this problem class" : "";
            for(auto* solution : candidates)
            {
                auto verdict = solution->admits(problem);
                if(!verdict.admits)
                {
                    if(reason.empty())
                    {
                        reason = verdict.reason;
                    }
                    continue;
                }
                if(solution->workspaceSize(problem) > context.workspaceSizeLimit)
                {
                    if(reason.empty())
                    {
                        reason = "every admitting kernel needs more workspace than the limit";
                    }
                    continue;
                }
                auto status = solution->prepare(problem, context, out);
                if(status == HIPTENSOR_STATUS_SUCCESS || !isDecline(status))
                {
                    return status;
                }
            }
            if(reason.empty())
            {
                reason = "no kernel admits the problem";
            }
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }

        hiptensorStatus_t IBackend::estimateWorkspace(Problem const&         problem,
                                                      EstimateContext const& context,
                                                      uint64_t&              estimate,
                                                      std::string&           reason) const
        {
            auto arch = supportsArch(problem.device);
            if(!arch.admits)
            {
                reason = arch.reason;
                return HIPTENSOR_STATUS_ARCH_MISMATCH;
            }

            std::vector<ISolution const*> candidates;
            enumerate(problem, candidates);
            rank(problem, candidates);

            // The first admitting candidate is the one a plan would take; MIN and MAX bound the
            // workspace over every admitting candidate.
            bool found = false;
            for(auto* solution : candidates)
            {
                if(!solution->admits(problem).admits)
                {
                    continue;
                }
                auto size = solution->workspaceSize(problem);
                if(!found)
                {
                    estimate = size;
                    found    = true;
                    if(context.workspacePref == HIPTENSOR_WORKSPACE_DEFAULT)
                    {
                        break;
                    }
                }
                else if(context.workspacePref == HIPTENSOR_WORKSPACE_MIN)
                {
                    estimate = std::min(estimate, size);
                }
                else
                {
                    estimate = std::max(estimate, size);
                }
            }
            if(!found)
            {
                reason = "no kernel admits the problem";
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }

        BackendRegistry::BackendRegistry()
        {
            add(ckbackend::makeCkBackend(), kCkPriority);
#ifdef HIPTENSOR_ENABLE_ROCKE_BACKEND
            add(rockebackend::makeRockeBackend(), kRockePriority);
#else
            static_cast<void>(kRockePriority);
#endif
        }

        BackendRegistry& BackendRegistry::instance()
        {
            static BackendRegistry registry;
            return registry;
        }

        void BackendRegistry::add(std::unique_ptr<IBackend> backend, int priority)
        {
            std::scoped_lock lock(mMutex);
            mEntries.push_back({std::move(backend), priority});
            std::stable_sort(mEntries.begin(), mEntries.end(), [](auto const& a, auto const& b) {
                return a.priority > b.priority;
            });
        }

        IBackend* BackendRegistry::find(std::string_view name) const
        {
            std::scoped_lock lock(mMutex);
            for(auto const& entry : mEntries)
            {
                if(entry.backend->name() == name)
                {
                    return entry.backend.get();
                }
            }
            return nullptr;
        }

        std::vector<IBackend*> BackendRegistry::backends() const
        {
            std::scoped_lock       lock(mMutex);
            std::vector<IBackend*> result;
            for(auto const& entry : mEntries)
            {
                result.push_back(entry.backend.get());
            }
            return result;
        }

        hiptensorStatus_t BackendPolicy::fromEnvironment(BackendPolicy& policy, std::string& error)
        {
            policy = BackendPolicy{};

            if(auto value = getEnvironmentVariable("HIPTENSOR_BACKEND"))
            {
                auto name = lowerTrimmed(*value);
                if(!name.empty() && name != "auto")
                {
                    if(!isKnownBackend(name))
                    {
                        error = "HIPTENSOR_BACKEND=" + *value + ": expected auto, ck or rocke";
                        return HIPTENSOR_STATUS_INVALID_VALUE;
                    }
                    policy.primary = name;
                }
            }

            if(auto value = getEnvironmentVariable("HIPTENSOR_BACKEND_FALLBACK"))
            {
                auto flag = lowerTrimmed(*value);
                if(flag == "1" || flag == "on" || flag == "true")
                {
                    policy.fallback = true;
                }
                else if(flag == "0" || flag == "off" || flag == "false")
                {
                    policy.fallback = false;
                }
                else if(!flag.empty())
                {
                    error = "HIPTENSOR_BACKEND_FALLBACK=" + *value + ": expected 1 or 0";
                    return HIPTENSOR_STATUS_INVALID_VALUE;
                }
            }

            if(auto value = getEnvironmentVariable("HIPTENSOR_BACKEND_AB"))
            {
                auto pair = lowerTrimmed(*value);
                if(!pair.empty())
                {
                    auto comma   = pair.find(',');
                    auto primary = lowerTrimmed(pair.substr(0, comma));
                    auto shadow  = comma == std::string::npos ? std::string()
                                                              : lowerTrimmed(pair.substr(comma + 1));
                    if(!isKnownBackend(primary) || !isKnownBackend(shadow))
                    {
                        error = "HIPTENSOR_BACKEND_AB=" + *value
                                + ": expected <primary>,<shadow> with backends ck or rocke";
                        return HIPTENSOR_STATUS_INVALID_VALUE;
                    }
                    policy.primary = primary;
                    policy.shadow  = shadow;
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }

        std::vector<std::string> BackendPolicy::order() const
        {
            std::vector<std::string> result{primary.empty() ? kDefaultBackend : primary};
            if(fallback && result.front() != kCkBackendName)
            {
                result.push_back(kCkBackendName);
            }
            return result;
        }

        hiptensorStatus_t createPlanBackend(hiptensorHandle_t handle,
                                            hiptensorPlan_t   plan,
                                            uint64_t          workspaceSizeLimit)
        {
            auto& logger = Logger::instance();
            char  msg[512];

            BackendPolicy policy;
            std::string   error;
            if(auto status = BackendPolicy::fromEnvironment(policy, error);
               status != HIPTENSOR_STATUS_SUCCESS)
            {
                logger->logError("hiptensorCreatePlan", error.c_str());
                return status;
            }

            auto state = std::make_unique<PlanBackendState>();
            if(auto status = makeProblem(*plan->mOpDesc, handle->getDevice(), state->problem);
               status != HIPTENSOR_STATUS_SUCCESS)
            {
                snprintf(msg,
                         sizeof(msg),
                         "The operation cannot be described to a backend: a tensor has more than "
                         "%d modes or an extent or stride that does not fit in 64 bits (%s)",
                         static_cast<int>(kMaxModes),
                         hiptensorGetErrorString(status));
                logger->logError("hiptensorCreatePlan", msg);
                return status;
            }

            PlanContext context{handle, plan, workspaceSizeLimit};
            std::string servedBy;
            auto        status
                = prepareFirst(policy.order(), state->problem, context, state->primary, servedBy);
            if(status != HIPTENSOR_STATUS_SUCCESS)
            {
                return status;
            }
            uint64_t requiredWorkspace = state->primary->workspaceSize();

            if(!policy.shadow.empty())
            {
                std::string shadowName;
                if(prepareFirst({policy.shadow}, state->problem, context, state->shadow, shadowName)
                   == HIPTENSOR_STATUS_SUCCESS)
                {
                    auto const& output = state->problem.output();
                    auto const  bytes  = static_cast<uint64_t>(output.spanElements())
                                       * hiptensorDataTypeSize(output.type);
                    if(hipMalloc(&state->shadowOutput.ptr, bytes) != hipSuccess)
                    {
                        state->shadowOutput.ptr = nullptr;
                        logger->logError("hiptensorCreatePlan",
                                         "A/B: cannot allocate the shadow output");
                        return HIPTENSOR_STATUS_ALLOC_FAILED;
                    }
                    state->shadowOutput.bytes = bytes;
                    requiredWorkspace = std::max(requiredWorkspace, state->shadow->workspaceSize());
                }
                else
                {
                    state->shadow.reset();
                    snprintf(msg,
                             sizeof(msg),
                             "A/B: the shadow backend %s does not serve this problem; the plan "
                             "runs %s only",
                             policy.shadow.c_str(),
                             servedBy.c_str());
                    logger->logHeuristics("hiptensorCreatePlan", msg);
                }
            }

            auto const& id = state->primary->kernelId();
            snprintf(msg,
                     sizeof(msg),
                     "Backend: %s, Op: %s, KernelId: %llu, KernelName: %s, Problem: %016llx",
                     servedBy.c_str(),
                     opKindName(state->problem.kind),
                     static_cast<unsigned long long>(id.uid),
                     id.specKey.c_str(),
                     static_cast<unsigned long long>(state->problem.digest()));
            logger->logHeuristics("hiptensorCreatePlan", msg);

            plan->mRequiredWorkspace = requiredWorkspace;
            plan->mBackend           = std::move(state);
            return HIPTENSOR_STATUS_SUCCESS;
        }

        hiptensorStatus_t executePlan(hiptensorHandle_t   handle,
                                      hiptensorPlan_t     plan,
                                      Scalars const&      scalars,
                                      Buffers const&      buffers,
                                      void*               workspace,
                                      uint64_t            workspaceSize,
                                      LaunchConfig const& cfg)
        {
            auto& state   = *plan->mBackend;
            auto  context = PlanContext{handle, plan, 0};

            // The shadow backend runs first, into its own buffer, so that both backends read the
            // same inputs even when the output aliases an input.
            bool shadowRan = false;
            if(state.shadow != nullptr && buffers.out != nullptr)
            {
                auto shadowBuffers = buffers;
                shadowBuffers.out  = state.shadowOutput.ptr;
                if(hipMemcpyAsync(state.shadowOutput.ptr,
                                  buffers.out,
                                  state.shadowOutput.bytes,
                                  hipMemcpyDeviceToDevice,
                                  cfg.stream)
                   == hipSuccess)
                {
                    auto status = state.shadow->run(
                        context, scalars, shadowBuffers, workspace, workspaceSize, cfg);
                    shadowRan = status == HIPTENSOR_STATUS_SUCCESS;
                    if(!shadowRan)
                    {
                        char msg[256];
                        snprintf(msg,
                                 sizeof(msg),
                                 "A/B: the shadow backend %s failed (%s)",
                                 state.shadow->kernelId().backend.c_str(),
                                 hiptensorGetErrorString(status));
                        Logger::instance()->logHeuristics("hiptensorExecute", msg);
                    }
                }
            }

            auto status
                = state.primary->run(context, scalars, buffers, workspace, workspaceSize, cfg);
            if(status == HIPTENSOR_STATUS_SUCCESS && shadowRan)
            {
                compareOutputs(state, buffers.out, cfg.stream);
            }
            return status;
        }

        hiptensorStatus_t estimateWorkspace(EstimateContext const& context, uint64_t& estimate)
        {
            auto& logger = Logger::instance();
            char  msg[512];

            BackendPolicy policy;
            std::string   error;
            if(auto status = BackendPolicy::fromEnvironment(policy, error);
               status != HIPTENSOR_STATUS_SUCCESS)
            {
                logger->logError("hiptensorEstimateWorkspaceSize", error.c_str());
                return status;
            }

            Problem problem;
            if(auto status = makeProblem(*context.desc, context.handle->getDevice(), problem);
               status != HIPTENSOR_STATUS_SUCCESS)
            {
                return status;
            }

            auto       status = HIPTENSOR_STATUS_NOT_SUPPORTED;
            auto const order  = policy.order();
            for(std::size_t i = 0; i < order.size(); i++)
            {
                auto*       backend = BackendRegistry::instance().find(order[i]);
                std::string reason  = "the backend is not built into this library";
                status              = HIPTENSOR_STATUS_NOT_SUPPORTED;
                if(backend != nullptr)
                {
                    status = backend->estimateWorkspace(problem, context, estimate, reason);
                }
                if(!isDecline(status))
                {
                    return status;
                }
                snprintf(msg,
                         sizeof(msg),
                         "Backend %s declined the %s problem: %s (%s)",
                         order[i].c_str(),
                         opKindName(problem.kind),
                         reason.c_str(),
                         hiptensorGetErrorString(status));
                logger->logHeuristics("hiptensorEstimateWorkspaceSize", msg);
            }
            return status;
        }

        hiptensorStatus_t writeKernelCache(hiptensorHandle_t handle, char const* fileName)
        {
            for(auto* backend : BackendRegistry::instance().backends())
            {
                if(backend->hasKernelCache())
                {
                    return backend->writeKernelCache(makeDeviceKey(handle->getDevice()), fileName);
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }

        hiptensorStatus_t readKernelCache(hiptensorHandle_t handle, char const* fileName)
        {
            for(auto* backend : BackendRegistry::instance().backends())
            {
                if(backend->hasKernelCache())
                {
                    return backend->readKernelCache(makeDeviceKey(handle->getDevice()), fileName);
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace backend
} // namespace hiptensor
