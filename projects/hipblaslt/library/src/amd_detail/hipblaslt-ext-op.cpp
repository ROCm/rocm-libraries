/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipblaslt/hipblaslt-ext-op.h"
#include "exceptions.hpp"
#include "hipblaslt-ext-op-internal.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hip/hip_ext.h>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <mutex>
#include <optional>
#include <rocblaslt-auxiliary.h>
#include <rocblaslt_secure_env.hpp>
#include <sstream>
#include <string>
#include <tensile_host.hpp>
#include <unordered_map>
#include <vector>

hipblasStatus_t hipblasltSoftmaxRun(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream);

hipblasStatus_t hipblasltExtSoftmax(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream)
try
{
    return hipblasltSoftmaxRun(datatype, m, n, dim, output, input, stream);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasltLayerNormRun(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream);

hipblasStatus_t hipblasltExtLayerNorm(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream)
try
{
    return hipblasltLayerNormRun(
        datatype, output, mean, invvar, input, m, n, eps, gamma, beta, stream);
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasltAMaxRun(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream);

hipblasStatus_t hipblasltExtAMax(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream)
try
{
    return hipblasltAMaxRun(datatype, outDatatype, output, input, m, n, stream);
}
catch(...)
{
    return exception_to_hipblas_status();
}

namespace
{
    constexpr uint32_t SUPPORTED_MAX_N = 256;
    constexpr uint32_t WORKGROUP_SIZE  = 256;

    // Strict per-base layout: every ExtOp .dat lives at
    //   <library_root>/<base-arch>/hipblasltExtOpLibrary_<base-arch>.dat
    // The HIPBLASLT_EXT_OP_LIBRARY_PATH env override is still honored verbatim
    // for explicit user-supplied paths. There is no flat fallback and no
    // hard-coded /opt/rocm path — a missing file logs an error and yields an
    // empty path, which surfaces as a clean ExtOpMasterLibrary load failure.
    std::string getExtOpLibraryPath(int deviceId)
    {
        // ROCM-26729 / SEC-00896: honor the override only for a non-privileged
        // process so a process in a secure execution context cannot be
        // redirected to an attacker-controlled ExtOp library via inherited
        // environment. Probe the privilege state once and reuse it for both the
        // lookup and the suppression diagnostic.
        const bool is_privileged = rocblaslt_process_is_privileged();

        if(auto libPath
           = rocblaslt_secure_getenv_impl("HIPBLASLT_EXT_OP_LIBRARY_PATH", is_privileged))
        {
            return libPath;
        }

        if(rocblaslt_env_suppressed_for_security_impl("HIPBLASLT_EXT_OP_LIBRARY_PATH",
                                                      is_privileged))
        {
            rocblaslt_log_error("getExtOpLibraryPath",
                                "HIPBLASLT_EXT_OP_LIBRARY_PATH",
                                "ignoring env override because the process is running in a "
                                "secure execution context (set-uid/set-gid or another "
                                "credential-changing exec, such as file capabilities); using "
                                "the default per-arch library location");
        }

        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, deviceId) != hipSuccess)
        {
            rocblaslt_log_error("getExtOpLibraryPath",
                                "hipGetDeviceProperties",
                                "failed to query device for ExtOp arch lookup");
            return {};
        }

        const std::string archName = rocblaslt_internal_get_arch_name(props);
        auto              basename = "hipblasltExtOpLibrary_" + archName;
        auto              relpath  = std::filesystem::path(archName) / (basename + ".dat");
        if(auto perArchPath = rocblaslt_find_library_relative_path(relpath))
            return perArchPath->string();

        auto relpath_gz = std::filesystem::path(archName) / (basename + ".dat.zlib");
        if(auto perArchPath = rocblaslt_find_library_relative_path(relpath_gz))
        {
            // Return the base .dat path; fileToMsgObject() probes for .zlib internally
            auto gz_path = perArchPath->string();
            return gz_path.substr(0, gz_path.size() - 5);
        }

        rocblaslt_log_error("getExtOpLibraryPath",
                            "rocblaslt_find_library_relative_path",
                            relpath.string().c_str());
        return {};
    }

    std::string hipDataTypeo_char(hipDataType type)
    {
        if(type == HIP_R_16F)
            return std::string("H");
        else if(type == HIP_R_32F)
            return std::string("S");
        return std::string("S");
    }

    inline uint32_t getSoftmaxNumWorkgroups(uint32_t m, uint32_t tileM)
    {
        return (m / tileM) + !!(m % tileM);
    }

    inline uint32_t getSoftmaxBestKernelTileN(uint32_t n)
    {
        const uint32_t exponent = std::ceil(std::log2(n));
        return 1 << exponent;
    }

    inline uint32_t getSoftmaxKernelTileM(uint32_t tileM)
    {
        return WORKGROUP_SIZE / tileM;
    }

    uint32_t elementNumBytes(hipDataType type)
    {
        if(type == HIP_R_16F)
            return 2;
        else if(type == HIP_R_32F)
            return 4;
        return 4;
    }

    inline uint32_t getLdsUsageByte(hipDataType datatype, uint32_t tileM, uint32_t tileN)
    {
        return elementNumBytes(datatype) * tileM * tileN;
    }

    // Each architecture has its own ExtOp library file. Devices of the same
    // architecture share the library loaded from it; libraries are never
    // unloaded. Throws if the file cannot be loaded.
    const hipblaslt_ext::ExtOpMasterLibrary& loadExtOpMasterLibrary(const std::string& path)
    {
        static std::mutex mutex;
        static std::unordered_map<std::string, std::unique_ptr<hipblaslt_ext::ExtOpMasterLibrary>>
            libraries;

        std::lock_guard<std::mutex> lock(mutex);
        auto&                       library = libraries[path];
        if(!library)
            library = std::make_unique<hipblaslt_ext::ExtOpMasterLibrary>(path);
        return *library;
    }

    // The ExtOp library of a device and an adapter loading code objects from
    // its folder, set up once on the device's first ExtOp call. A device whose
    // library fails to load, or holds no kernels for its architecture, keeps a
    // null library and is not retried.
    struct ExtOpDevice
    {
        std::once_flag                                     once;
        const hipblaslt_ext::ExtOpMasterLibrary*           library = nullptr;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::string                                        archName;
        std::shared_ptr<TensileLite::Hardware>             hardware;
    };

    ExtOpDevice& extOpDevice(int deviceId)
    {
        static std::vector<ExtOpDevice> devices = [] {
            int numDevices{};
            if(hipGetDeviceCount(&numDevices) != hipSuccess)
                numDevices = 0;
            return std::vector<ExtOpDevice>(numDevices);
        }();

        auto& device = devices.at(deviceId);
        std::call_once(device.once, [&] {
            device.adapter  = std::make_unique<TensileLite::hip::SolutionAdapter>();
            const auto path = getExtOpLibraryPath(deviceId);
            try
            {
                hipDeviceProp_t props{};
                if(hipGetDeviceProperties(&props, deviceId) != hipSuccess)
                {
                    rocblaslt_log_error("extOpDevice", "hipGetDeviceProperties", path.c_str());
                    return;
                }
                device.archName     = rocblaslt_internal_get_arch_name(props);
                device.hardware     = TensileLite::hip::GetDevice(props, deviceId);
                const auto& library = loadExtOpMasterLibrary(path);
                if(!library.hasArchitecture(device.archName))
                {
                    rocblaslt_log_error(
                        "extOpDevice", path.c_str(), ("no kernels for " + device.archName).c_str());
                    return;
                }
                device.adapter->codeObjectDir(library.getLibraryFolder());
                device.library = &library;
            }
            catch(const std::exception& e)
            {
                rocblaslt_log_error("extOpDevice", path.c_str(), e.what());
            }
        });
        return device;
    }
}

hipblasStatus_t hipblasltSoftmaxRun(hipDataType datatype,
                                    uint32_t    m,
                                    uint32_t    n,
                                    uint32_t    dim,
                                    void*       output,
                                    void*       input,
                                    hipStream_t stream)
{
    if(datatype != HIP_R_32F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(dim != 1)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    const auto tileN = getSoftmaxBestKernelTileN(n);
    const auto tileM = getSoftmaxKernelTileM(tileN);

    if(tileN > SUPPORTED_MAX_N)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    if(err != hipSuccess)
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    auto& extOp = extOpDevice(currentDeviceId);
    if(!extOp.library)
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    auto*       adapter   = extOp.adapter.get();
    const auto& gpu       = extOp.hardware;
    const auto& archName  = extOp.archName;
    auto&       masterLib = *extOp.library;
    const auto& lib       = masterLib
                          .getLibrary(archName,
                                      hipblaslt_ext::SoftmaxSolutionLibrary::opName,
                                      hipDataTypeo_char(datatype))
                          ->as<hipblaslt_ext::SoftmaxSolutionLibrary>();
    auto sol = lib.findBestSolution(
        hipblaslt_ext::SoftmaxProblem(m, n, hipDataType_to_tensile_type(datatype)), *gpu);
    const auto kernelName = sol->name();
    err                   = adapter->initKernel(kernelName);
    TensileLite::KernelArguments kArgs(false);
    kArgs.append("input", input);
    kArgs.append("output", output);
    kArgs.append("m", m);
    kArgs.append("n", n);
    const auto                    numWorkgroups = getSoftmaxNumWorkgroups(m, tileM);
    TensileLite::KernelInvocation invocation{kernelName,
                                             sol->getCodeObjectPath(),
                                             false,
                                             {1, 1, 1},
                                             {WORKGROUP_SIZE, 1, 1},
                                             {numWorkgroups, 1, 1},
                                             {numWorkgroups * WORKGROUP_SIZE, 1, 1},
                                             getLdsUsageByte(datatype, tileM, tileN),
                                             kArgs};

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasltLayerNormRun(hipDataType datatype,
                                      void*       output,
                                      void*       mean,
                                      void*       invvar,
                                      void*       input,
                                      uint32_t    m,
                                      uint32_t    n,
                                      float       eps,
                                      void*       gamma,
                                      void*       beta,
                                      hipStream_t stream)
{
    if(datatype != HIP_R_32F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(output == nullptr || mean == nullptr || invvar == nullptr || input == nullptr || m == 0
       || n == 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    if(err != hipSuccess)
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    auto& extOp = extOpDevice(currentDeviceId);
    if(!extOp.library)
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    auto*       adapter   = extOp.adapter.get();
    const auto& gpu       = extOp.hardware;
    const auto& archName  = extOp.archName;
    auto&       masterLib = *extOp.library;
    const auto& lib       = masterLib
                          .getLibrary(archName,
                                      hipblaslt_ext::LayerNormSolutionLibrary::opName,
                                      hipDataTypeo_char(datatype))
                          ->as<hipblaslt_ext::LayerNormSolutionLibrary>();
    auto sol = lib.findBestSolution(
        hipblaslt_ext::LayerNormProblem(m, n, hipDataType_to_tensile_type(datatype)), *gpu);
    const auto kernelName    = sol->name();
    err                      = adapter->initKernel(kernelName);
    const auto numWorkgroups = m;

    TensileLite::KernelInvocation invocation;
    invocation.kernelName      = kernelName;
    invocation.codeObjectFile  = sol->getCodeObjectPath();
    invocation.workGroupSize.x = sol->getNumWorkitems();
    invocation.workGroupSize.y = 1;
    invocation.workGroupSize.z = 1;
    invocation.numWorkGroups.x = 1;
    invocation.numWorkGroups.y = numWorkgroups;
    invocation.numWorkGroups.z = 1;
    invocation.numWorkItems.x  = sol->getNumWorkitems();
    invocation.numWorkItems.y  = numWorkgroups;
    invocation.numWorkItems.z  = 1;
    invocation.sharedMemBytes  = 32 * sizeof(float);
    invocation.args            = TensileLite::KernelArguments(false);
    invocation.args.reserve(60, 9);
    invocation.args.append("output", output);
    invocation.args.append("mean", mean);
    invocation.args.append("invvar", invvar);
    invocation.args.append("input", input);
    invocation.args.append("gamma", gamma);
    invocation.args.append("beta", beta);
    invocation.args.append("m", m);
    invocation.args.append("n", n);
    invocation.args.append("eps", eps);

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasltAMaxRun(const hipDataType datatype,
                                 const hipDataType outDatatype,
                                 void*             output,
                                 void*             input,
                                 uint32_t          m,
                                 uint32_t          n,
                                 hipStream_t       stream)
{
    if(datatype != HIP_R_32F && datatype != HIP_R_16F)
    {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if(output == nullptr || input == nullptr || m == 0 || n == 0)
        return HIPBLAS_STATUS_INVALID_VALUE;

    uint32_t    len = m * n;
    int         currentDeviceId{};
    auto        err       = hipGetDevice(&currentDeviceId);
    if(err != hipSuccess)
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    auto& extOp = extOpDevice(currentDeviceId);
    if(!extOp.library)
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    auto*       adapter   = extOp.adapter.get();
    const auto& gpu       = extOp.hardware;
    const auto& archName  = extOp.archName;
    auto&       masterLib = *extOp.library;
    const auto& lib       = masterLib
                          .getLibrary(archName,
                                      hipblaslt_ext::AMaxSolutionLibrary::opName,
                                      hipDataTypeo_char(datatype))
                          ->as<hipblaslt_ext::AMaxSolutionLibrary>();
    auto sol = lib.findBestSolution(
        hipblaslt_ext::AMaxProblem(
            len, hipDataType_to_tensile_type(datatype), hipDataType_to_tensile_type(outDatatype)),
        *gpu);
    const auto kernelName = sol->name();
    err                   = adapter->initKernel(kernelName);

    TensileLite::KernelInvocation invocation;
    invocation.kernelName      = kernelName;
    invocation.codeObjectFile  = sol->getCodeObjectPath();
    invocation.clusterDim.x    = 1;
    invocation.clusterDim.y    = 1;
    invocation.clusterDim.z    = 1;
    invocation.workGroupSize.x = sol->getNumWorkitems();
    invocation.workGroupSize.y = 1;
    invocation.workGroupSize.z = 1;
    invocation.numWorkGroups.x = 1;
    invocation.numWorkGroups.y = 1;
    invocation.numWorkGroups.z = 1;
    invocation.numWorkItems.x  = sol->getNumWorkitems();
    invocation.numWorkItems.y  = 1;
    invocation.numWorkItems.z  = 1;
    invocation.sharedMemBytes  = 32 * sizeof(float);
    invocation.args            = TensileLite::KernelArguments(false);
    invocation.args.reserve(20, 3);
    invocation.args.append("output", output);
    invocation.args.append("input", input);
    invocation.args.append("length", len);

    err = adapter->launchKernel(invocation, stream, nullptr, nullptr);

    if(err)
    {
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }

    return HIPBLAS_STATUS_SUCCESS;
}

// Setter for the members of class UserClientArguments, only for internal use.
// The class UserClientArguments is declared and defined in utility.hpp and utility.cpp respectively
void hipblasltSetFlushValue(bool newFlush)
{
    UserClientArguments clientArguments;
    clientArguments.SetFlushValue(newFlush);
}
void hipblasltSetRotatingBufferSizeValue(int newrotatingBufferSize)
{
    UserClientArguments clientArguments;
    clientArguments.SetRotatingBufferSizeValue(newrotatingBufferSize);
}

void hipblasltSetColdIterationsValue(int newColdIterations)
{
    UserClientArguments clientArguments;
    clientArguments.SetColdIterationsValue(newColdIterations);
}

void hipblasltSetHotIterationsValue(int newHotIterations)
{
    UserClientArguments clientArguments;
    clientArguments.SetHotIterationsValue(newHotIterations);
}

// Get hipblaslt client performance args, For internal use only.
double hipblasltGetTotalGranularityValue()
{
    return hipblasltClientPerformanceArgs::totalGranularity;
}

double hipblasltGetTilesPerCuValue()
{
    return hipblasltClientPerformanceArgs::tilesPerCu;
}

double hipblasltGetTile0Granularity()
{
    return hipblasltClientPerformanceArgs::tile0Granularity;
}
double hipblasltGetTile1Granularity()
{
    return hipblasltClientPerformanceArgs::tile1Granularity;
}
double hipblasltGetCuGranularity()
{
    return hipblasltClientPerformanceArgs::cuGranularity;
}
double hipblasltGetWaveGranularity()
{
    return hipblasltClientPerformanceArgs::waveGranularity;
}
int hipblasltGetCUs()
{
    return hipblasltClientPerformanceArgs::CUs;
}
size_t hipblasltGetMemWriteBytesD()
{
    return hipblasltClientPerformanceArgs::memWriteBytesD;
}
size_t hipblasltGetMemReadBytes()
{
    return hipblasltClientPerformanceArgs::memReadBytes;
}
