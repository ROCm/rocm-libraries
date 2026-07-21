// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    F8Mode getDefaultF8ModeForCurrentHipDevice()
    {
        auto const& arch = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();
        return arch.HasCapability(GPUCapability::HasNaNoo) ? F8Mode::NaNoo : F8Mode::OCP;
    }

    bool getDefaultValueForKernelGraphDOTSerialization()
    {
        return Settings::getInstance()->get(Settings::SaveAssembly);
    }
}
