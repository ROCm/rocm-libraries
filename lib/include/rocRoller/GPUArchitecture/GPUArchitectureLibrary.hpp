
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
    const std::string ARCHITECTURE_YAML_FILE     = "source/rocRoller/GPUArchitecture_def.yaml";
    const std::string ARCHITECTURE_MSGPACK_FILE  = "source/rocRoller/GPUArchitecture_def.msgpack";
    const std::string ENV_ARCHITECTURE_YAML_FILE = "ROCROLLER_ARCHITECTURE_YAML_FILE";

    class GPUArchitectureLibrary
    {
    public:
        static bool                          HasCapability(GPUArchitectureTarget, GPUCapability);
        static int                           GetCapability(GPUArchitectureTarget, GPUCapability);
        static bool                          HasCapability(std::string, GPUCapability);
        static int                           GetCapability(std::string, GPUCapability);
        static bool                          HasCapability(std::string, std::string);
        static int                           GetCapability(std::string, std::string);
        static rocRoller::GPUInstructionInfo GetInstructionInfo(GPUArchitectureTarget, std::string);
        static rocRoller::GPUInstructionInfo GetInstructionInfo(std::string, std::string);

        static GPUArchitecture GetArch(GPUArchitectureTarget const& target);
        static GPUArchitecture GetArch(std::string const& archName);

        static GPUArchitecture GetHipDeviceArch(int deviceIdx);
        static GPUArchitecture GetDefaultHipDeviceArch(int& deviceIdx);
        static GPUArchitecture GetDefaultHipDeviceArch();

        static void GetCurrentHipDevices(std::vector<GPUArchitecture>&, int&);

        static void            GetCurrentDevices(std::vector<GPUArchitecture>&, int&);
        static GPUArchitecture GetDevice(std::string);
        /**
         * @brief Returns a vector of strings listing all of the supported ISAs
         *
         * @return std::vector<std::string>
         */
        static std::vector<std::string> getAllSupportedISAs();

        static std::map<GPUArchitectureTarget, GPUArchitecture> LoadLibrary();

    private:
        static std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectures;
    };
}

#include "GPUArchitectureLibrary_impl.hpp"
