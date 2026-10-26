// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "CodeObjectLoad.hpp"

#include "Kernel.hpp"
#include "KpackModule.hpp"
#include "KpackProgram.hpp"
#include "device/ScopedDevice.hpp"

#include <utility>

namespace hip_kernel_provider::compilation
{

LoadedCodeObject loadCodeObjectOnDevice(const std::vector<uint8_t>& bytes,
                                        const std::string& symbol,
                                        int deviceOrdinal,
                                        const std::string& label)
{
    // A module belongs to the device that is current at hipModuleLoadData. Make the
    // target device current first.
    const device::ScopedDevice binding(deviceOrdinal);
    if(!binding.bound())
    {
        throw CodeObjectLoadFailure(CodeObjectLoadStage::BIND_DEVICE,
                                    hipErrorInvalidDevice,
                                    "cannot make device " + std::to_string(deviceOrdinal)
                                        + " current to load the code object of " + label);
    }

    // hipModuleLoadData takes no length, so it must not receive an empty buffer.
    if(bytes.empty())
    {
        throw CodeObjectLoadFailure(CodeObjectLoadStage::MODULE_LOAD,
                                    hipErrorInvalidImage,
                                    "hipModuleLoadData was not called for " + label
                                        + ": the code object is empty");
    }

    hipModule_t module = nullptr;
    hipError_t status = hipModuleLoadData(&module, bytes.data());
    if(status != hipSuccess)
    {
        throw CodeObjectLoadFailure(CodeObjectLoadStage::MODULE_LOAD,
                                    status,
                                    "hipModuleLoadData rejected the code object of " + label
                                        + " on device " + std::to_string(deviceOrdinal) + ": "
                                        + hipGetErrorString(status));
    }

    // Owned before the symbol lookup, so a failed lookup unloads the module on its device.
    auto owner = std::make_shared<const KpackModule>(module, deviceOrdinal);

    hipFunction_t function = nullptr;
    status = hipModuleGetFunction(&function, module, symbol.c_str());
    if(status != hipSuccess)
    {
        throw CodeObjectLoadFailure(CodeObjectLoadStage::SYMBOL_LOOKUP,
                                    status,
                                    "hipModuleGetFunction found no symbol '" + symbol
                                        + "' in the code object of " + label + ": "
                                        + hipGetErrorString(status));
    }

    LoadedCodeObject loaded;
    loaded.program = std::make_unique<KpackProgram>(std::move(owner), label);
    loaded.kernel = std::make_unique<Kernel>(function, symbol, deviceOrdinal);
    return loaded;
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
