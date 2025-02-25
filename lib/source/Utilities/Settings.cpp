#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

namespace rocRoller
{
    F8Mode getDefaultF8ModeForCurrentHipDevice()
    {
        const auto arch = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch();
        return arch.HasCapability(GPUCapability::HasNaNoo) ? F8Mode::NaNoo : F8Mode::OCP;
    }
}
