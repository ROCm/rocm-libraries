
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
        bool HasCapability(GPUArchitectureTarget const&, GPUCapability const&);
        int  GetCapability(GPUArchitectureTarget const&, GPUCapability const&);
        bool HasCapability(std::string const&, GPUCapability const&);
        int  GetCapability(std::string const&, GPUCapability const&);
        bool HasCapability(std::string const&, std::string const&);
        int  GetCapability(std::string const&, std::string const&);
        rocRoller::GPUInstructionInfo GetInstructionInfo(GPUArchitectureTarget const&,
                                                         std::string const&);
        rocRoller::GPUInstructionInfo GetInstructionInfo(std::string const&, std::string const&);

        GPUArchitecture GetArch(GPUArchitectureTarget const& target);
        GPUArchitecture GetArch(std::string const& archName);

        GPUArchitecture GetHipDeviceArch(int deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch(int& deviceIdx);
        GPUArchitecture GetDefaultHipDeviceArch();

        void GetCurrentHipDevices(std::vector<GPUArchitecture>&, int&);

        void            GetCurrentDevices(std::vector<GPUArchitecture>&, int&);
        GPUArchitecture GetDevice(std::string const&);
        /**
         * @brief Returns a vector of strings listing all of the supported ISAs
         *
         * @return std::vector<std::string>
         */
        std::vector<std::string> getAllSupportedISAs();
        std::vector<std::string> getMFMASupportedISAs();

        std::map<GPUArchitectureTarget, GPUArchitecture> LoadLibrary();

        GPUArchitectureLibrary()
            : GPUArchitectures(LoadLibrary())
        {
        }

    private:
        std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectures;
    };
}

#include "GPUArchitectureLibrary_impl.hpp"
