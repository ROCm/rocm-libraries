// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipModule.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <sstream>
#include <string>
#include <utility>

namespace ck_dsl_provider {

namespace {

[[noreturn]] void throwHipError(hipError_t err, std::string_view context) {
    const char* name = hipGetErrorName(err);
    const char* msg = hipGetErrorString(err);
    std::ostringstream oss;
    oss << context << ": " << (name != nullptr ? name : "hipError(unknown)") << ": "
        << (msg != nullptr ? msg : "no error string available")
        << " (code=" << static_cast<int>(err) << ")";
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, oss.str());
}

void checkHip(hipError_t err, std::string_view context) {
    if (err != hipSuccess) {
        throwHipError(err, context);
    }
}

}  // namespace

HipModule::HipModule(const KernelArtifact& artifact) : _kernelName(artifact.kernelName) {
    if (artifact.hsaco.empty()) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "HipModule: refusing to load empty HSACO blob for kernel '" + _kernelName + "'");
    }
    if (_kernelName.empty()) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "HipModule: KernelArtifact has empty kernelName");
    }

    checkHip(hipModuleLoadData(&_module, artifact.hsaco.data()),
             "HipModule::ctor hipModuleLoadData for '" + _kernelName + "'");

    hipError_t funcErr = hipModuleGetFunction(&_function, _module, _kernelName.c_str());
    if (funcErr != hipSuccess) {
        // Module is already loaded; unload it before propagating so we
        // don't leak the module handle when the symbol lookup fails.
        // Best-effort: if unload itself fails (e.g. driver wedged), we
        // log and continue -- the original error is the one the caller
        // needs to see.
        hipError_t unloadErr = hipModuleUnload(_module);
        _module = nullptr;
        if (unloadErr != hipSuccess) {
            try {
                HIPDNN_PLUGIN_LOG_INFO(
                    "HipModule: hipModuleUnload during cleanup also failed: code="
                    << static_cast<int>(unloadErr));
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
        }
        throwHipError(funcErr, "HipModule::ctor hipModuleGetFunction for '" + _kernelName + "'");
    }
}

HipModule::~HipModule() noexcept {
    if (_module != nullptr) {
        hipError_t err = hipModuleUnload(_module);
        if (err != hipSuccess) {
            // Destructor is noexcept; log and swallow. Realistically
            // the only way unload fails here is a driver-level fault
            // during process teardown, where there is nothing the
            // plugin can usefully do.
            try {
                HIPDNN_PLUGIN_LOG_INFO("HipModule::dtor hipModuleUnload for '"
                                       << _kernelName
                                       << "' failed: code=" << static_cast<int>(err));
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
        }
        _module = nullptr;
        _function = nullptr;
    }
}

HipModule::HipModule(HipModule&& other) noexcept
    : _module(other._module),
      _function(other._function),
      _kernelName(std::move(other._kernelName)) {
    other._module = nullptr;
    other._function = nullptr;
}

void HipModule::launch(const std::vector<std::byte>& packedArgs,
                       const KernelArtifact::GridSpec& grid, const KernelArtifact::BlockSpec& block,
                       std::uint32_t ldsBytes, hipStream_t stream) {
    // hipModuleLaunchKernel's BUFFER_POINTER path takes a non-const
    // void* into the args buffer; HIP only reads from it during the
    // call, so const_cast'ing the local buffer's data() is safe.
    std::size_t argsSize = packedArgs.size();
    void* argsPtr = const_cast<void*>(static_cast<const void*>(packedArgs.data()));

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, argsPtr, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize, HIP_LAUNCH_PARAM_END};

    // For kernels that take no parameters, pass nullptr config -- HIP
    // rejects an extras array with BUFFER_SIZE==0 on some driver
    // versions. The two args (kernelParams, extra) are mutually
    // exclusive per the HIP runtime contract.
    void** extras = packedArgs.empty() ? nullptr : config;

    hipError_t err = hipModuleLaunchKernel(_function, grid.x, grid.y, grid.z, block.x, block.y,
                                           block.z, ldsBytes, stream,
                                           /*kernelParams=*/nullptr, extras);
    if (err != hipSuccess) {
        throwHipError(err, "HipModule::launch hipModuleLaunchKernel for '" + _kernelName + "'");
    }
}

}  // namespace ck_dsl_provider
