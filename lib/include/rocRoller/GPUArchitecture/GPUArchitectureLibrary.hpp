
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "GPUArchitecture.hpp"
#include "GPUArchitectureTarget.hpp"
#include "GPUCapability.hpp"
#include "GPUInstructionInfo.hpp"
#include <rocRoller/Utilities/LazySingleton.hpp>

namespace rocRoller
{
    class GPUArchitectureLibrary : public LazySingleton<GPUArchitectureLibrary>
    {
    public:
        bool               HasCapability(GPUArchitectureTarget const&, GPUCapability const&);
        bool               HasCapability(GPUArchitectureTarget const&, std::string const&);
        int                GetCapability(GPUArchitectureTarget const&, GPUCapability const&);
        GPUInstructionInfo GetInstructionInfo(GPUArchitectureTarget const&, std::string const&);

        GPUArchitecture GetArch(GPUArchitectureTarget const& target);

        GPUArchitecture GetHipDeviceArch(int deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch(int& deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch();
        bool            HasHipDevice();

        void GetCurrentDevices(std::vector<GPUArchitecture>&, int&);

        std::vector<GPUArchitectureTarget> getAllSupportedISAs();
        std::vector<GPUArchitectureTarget> getCDNAISAs();
        std::vector<GPUArchitectureTarget> getMFMASupportedISAs();

        std::map<GPUArchitectureTarget, GPUArchitecture> LoadLibrary();

        GPUArchitectureLibrary()
            : m_gpuArchitectures(LoadLibrary())
        {
        }

    private:
        std::map<GPUArchitectureTarget, GPUArchitecture> m_gpuArchitectures;
    };
}

#include "GPUArchitectureLibrary_impl.hpp"
