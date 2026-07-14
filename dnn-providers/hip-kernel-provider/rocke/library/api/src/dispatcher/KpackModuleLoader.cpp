// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/KpackModuleLoader.hpp"

namespace rocke_client::dispatcher
{

KpackLoadResult loadKernelFromKpack(const std::string& kpackPath,
                                    const std::string& tocKey,
                                    const std::string& arch,
                                    const std::string& kernelName)
{
    KpackLoadResult result;

    // --- Phase 1: open kpack archive ---
    kpack_archive_t archive = nullptr;
    result.kpackError = kpack_open(kpackPath.c_str(), &archive);
    if(result.kpackError != KPACK_SUCCESS)
    {
        return result;
    }

    // --- Phase 2: extract HSACO bytes ---
    // kpack_get_kernel allocates a copy; caller must free with kpack_free_kernel.
    void* kernelData = nullptr;
    size_t kernelSize = 0;
    result.kpackError
        = kpack_get_kernel(archive, tocKey.c_str(), arch.c_str(), &kernelData, &kernelSize);
    if(result.kpackError != KPACK_SUCCESS)
    {
        kpack_close(archive);
        return result;
    }

    // --- Phase 3: load HSACO into a HIP module ---
    // hipModuleLoadData copies the image; kpack buffer may be freed immediately after.
    result.hipError = hipModuleLoadData(&result.module, kernelData);
    kpack_free_kernel(kernelData);
    kpack_close(archive);

    if(result.hipError != hipSuccess)
    {
        return result;
    }

    // --- Phase 4: look up the kernel function ---
    result.hipError = hipModuleGetFunction(&result.fn, result.module, kernelName.c_str());
    if(result.hipError != hipSuccess)
    {
        static_cast<void>(hipModuleUnload(result.module));
        result.module = nullptr;
    }

    return result;
}

} // namespace rocke_client::dispatcher
