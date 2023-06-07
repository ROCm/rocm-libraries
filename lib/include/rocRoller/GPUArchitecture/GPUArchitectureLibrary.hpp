
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

namespace rocRoller
{
    class GPUArchitectureLibrary
    {
    public:
        static bool HasCapability(GPUArchitectureTarget const&, GPUCapability const&);
        static int  GetCapability(GPUArchitectureTarget const&, GPUCapability const&);
        static bool HasCapability(std::string const&, GPUCapability const&);
        static int  GetCapability(std::string const&, GPUCapability const&);
        static bool HasCapability(std::string const&, std::string const&);
        static int  GetCapability(std::string const&, std::string const&);
        static rocRoller::GPUInstructionInfo GetInstructionInfo(GPUArchitectureTarget const&,
                                                                std::string const&);
        static rocRoller::GPUInstructionInfo GetInstructionInfo(std::string const&,
                                                                std::string const&);

        static GPUArchitecture GetArch(GPUArchitectureTarget const& target);
        static GPUArchitecture GetArch(std::string const& archName);

        static GPUArchitecture GetHipDeviceArch(int deviceIdx);
        static GPUArchitecture GetDefaultHipDeviceArch(int& deviceIdx);
        static GPUArchitecture GetDefaultHipDeviceArch();

        static void GetCurrentHipDevices(std::vector<GPUArchitecture>&, int&);

        static void            GetCurrentDevices(std::vector<GPUArchitecture>&, int&);
        static GPUArchitecture GetDevice(std::string const&);
        /**
         * @brief Returns a vector of strings listing all of the supported ISAs
         *
         * @return std::vector<std::string>
         */
        static std::vector<std::string> getAllSupportedISAs();
        static std::vector<std::string> getMFMASupportedISAs();

        static std::map<GPUArchitectureTarget, GPUArchitecture> LoadLibrary();

    private:
        static std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectures;
    };
}

#include "GPUArchitectureLibrary_impl.hpp"
