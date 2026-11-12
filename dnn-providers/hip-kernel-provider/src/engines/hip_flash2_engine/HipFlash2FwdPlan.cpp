// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipFlash2FwdPlan.hpp"
#include "device/LaunchDeviceOrdinal.hpp"
#include "device/ScopedDevice.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace hip_flash2_engine
{

HipFlash2FwdPlan::HipFlash2FwdPlan(std::unique_ptr<Flash2Kernels> kernels, Flash2FwdParams params)
    : _kernels(std::move(kernels))
    , _params(std::move(params))
{
}

size_t HipFlash2FwdPlan::getWorkspaceSize(const Handle& /*handle*/) const
{
    // Flash-Attention 2 V7 uses only registers and LDS -- zero global workspace.
    return 0;
}

void HipFlash2FwdPlan::execute(const Handle& handle,
                               const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                               uint32_t numDeviceBuffers,
                               void* /*workspace*/) const
{
    // -- 1. Build UID -> device pointer map ------------------------------------
    std::unordered_map<int64_t, void*> uidToPtrMap;
    uidToPtrMap.reserve(numDeviceBuffers);
    for(uint32_t i = 0; i < numDeviceBuffers; ++i)
    {
        uidToPtrMap[deviceBuffers[i].uid] = deviceBuffers[i].ptr;
    }

    auto findPtr = [&](int64_t uid, const char* name) -> void* {
        auto it = uidToPtrMap.find(uid);
        if(it == uidToPtrMap.end())
        {
            HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- missing buffer for tensor '"
                                    << name << "' (uid=" << uid << ")");
            throw std::runtime_error(std::string("HipFlash2FwdPlan: missing tensor buffer '") + name
                                     + "'");
        }
        return it->second;
    };

    void* q = findPtr(_params.qUid, "Q");
    void* k = findPtr(_params.kUid, "K");
    void* v = findPtr(_params.vUid, "V");
    void* o = findPtr(_params.oUid, "O");

    // -- 2. Populate kernel argument struct -----------------------------------
    Flash2KernelArgs args{};
    args.ptrQ = q;
    args.ptrK = k;
    args.ptrV = v;
    args.ptrO = o;

    args.batch = _params.batch;
    args.numHeadsQ = _params.numHeadsQ;
    args.numHeadsK = _params.numHeadsK;
    args.seqLenQ = _params.seqLenQ;
    args.seqLenKv = _params.seqLenKv;
    args.headDim = _params.headDim;
    args.causal = _params.causal ? 1 : 0;

    args.scale = _params.attnScale;

    // Strides (in elements, BHSD layout).
    // Guard against int64_t -> int truncation (I9): strides must fit in int.
    // For the FP16 shapes this engine accepts (seq <= 131072, D <= 128, H <= 128,
    // B <= 32768) the largest possible batch stride is ~32768x128x131072x128
    // which overflows int.  Log and abort if any stride exceeds INT_MAX.
    auto checkedStride = [&](int64_t s, const char* name) -> int {
        if(s > static_cast<int64_t>(std::numeric_limits<int>::max()) || s < 0)
        {
            HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- stride '" << name << "'=" << s
                                                                            << " out of int range");
            throw std::overflow_error(std::string("HipFlash2FwdPlan: stride overflow '") + name
                                      + "'");
        }
        return static_cast<int>(s);
    };
    args.qStrideBatch = checkedStride(_params.qStrideBatch, "qStrideBatch");
    args.qStrideHead = checkedStride(_params.qStrideHead, "qStrideHead");
    args.qStrideSeq = checkedStride(_params.qStrideSeq, "qStrideSeq");
    args.kStrideBatch = checkedStride(_params.kStrideBatch, "kStrideBatch");
    args.kStrideHead = checkedStride(_params.kStrideHead, "kStrideHead");
    args.kStrideSeq = checkedStride(_params.kStrideSeq, "kStrideSeq");
    args.vStrideBatch = checkedStride(_params.vStrideBatch, "vStrideBatch");
    args.vStrideHead = checkedStride(_params.vStrideHead, "vStrideHead");
    args.vStrideSeq = checkedStride(_params.vStrideSeq, "vStrideSeq");
    args.oStrideBatch = checkedStride(_params.oStrideBatch, "oStrideBatch");
    args.oStrideHead = checkedStride(_params.oStrideHead, "oStrideHead");
    args.oStrideSeq = checkedStride(_params.oStrideSeq, "oStrideSeq");

    // -- 3. Grid dimensions ----------------------------------------------------
    // Tile size is a property of the SELECTED variant, not a constant.
    const unsigned int qPerCta = _params.qPerCta;
    const unsigned int gridX = (static_cast<unsigned>(_params.seqLenQ) + qPerCta - 1u) / qPerCta;
    // Finding 3 fix: kernel decodes blockIdx.y=batch, blockIdx.z=head_q
    const auto gridY = static_cast<unsigned>(_params.batch);
    const auto gridZ = static_cast<unsigned>(_params.numHeadsQ);

    // Block dim must match the selected variant's __launch_bounds__. A
    // mismatch is not benign: too few threads silently computes wrong results,
    // too many fails with hipErrorLaunchFailure (719).
    const unsigned int blockDim = _params.blockDim;

    // -- 4. Dispatch -----------------------------------------------------------
    // The module is the one loaded on the device the handle's stream names, not the
    // device the plan was built under: a plan may be executed under any handle.
    const int deviceOrdinal = hip_kernel_provider::device::launchDeviceOrdinal(handle.getStream());
    const Flash2Module kernel = _kernels->forOrdinal(deviceOrdinal);

    // A module has to be launched from the device it was loaded on.
    const hip_kernel_provider::device::ScopedDevice binding(deviceOrdinal);
    if(!binding.bound())
    {
        const std::string msg = "HipFlash2FwdPlan::execute: cannot make device "
                                + std::to_string(deviceOrdinal) + " current to launch";
        HIPDNN_PLUGIN_LOG_ERROR(msg);
        throw std::runtime_error(msg);
    }

    // I5: propagate launch failure so callers see a hard error.
    const bool ok = launchFlash2Kernel(
        kernel->function(), args, gridX, gridY, gridZ, blockDim, handle.getStream());
    if(!ok)
    {
        HIPDNN_PLUGIN_LOG_ERROR("HipFlash2FwdPlan::execute -- kernel launch failed");
        throw std::runtime_error("HipFlash2FwdPlan::execute: hipModuleLaunchKernel failed");
    }
}

} // namespace hip_flash2_engine
