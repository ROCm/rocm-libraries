#pragma once
#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace rocRoller
{
    inline bool GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget isaVersion,
                                                      GPUCapability         capability)
    {
        return GPUArchitectureLibrary::GPUArchitectures.find(isaVersion)
                   != GPUArchitectureLibrary::GPUArchitectures.end()
               && GPUArchitectureLibrary::GPUArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline bool GPUArchitectureLibrary::HasCapability(std::string   isaVersionString,
                                                      GPUCapability capability)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.find(isaVersion)
                   != GPUArchitectureLibrary::GPUArchitectures.end()
               && GPUArchitectureLibrary::GPUArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline bool GPUArchitectureLibrary::HasCapability(std::string isaVersionString,
                                                      std::string capabilityString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.find(isaVersion)
                   != GPUArchitectureLibrary::GPUArchitectures.end()
               && GPUArchitectureLibrary::GPUArchitectures.at(isaVersion)
                      .HasCapability(capabilityString);
    }

    inline int GPUArchitectureLibrary::GetCapability(GPUArchitectureTarget isaVersion,
                                                     GPUCapability         capability)
    {
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion).GetCapability(capability);
    }

    inline int GPUArchitectureLibrary::GetCapability(std::string   isaVersionString,
                                                     GPUCapability capability)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion).GetCapability(capability);
    }

    inline int GPUArchitectureLibrary::GetCapability(std::string isaVersionString,
                                                     std::string capabilityString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion)
            .GetCapability(capabilityString);
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitectureLibrary::GetInstructionInfo(GPUArchitectureTarget isaVersion,
                                                   std::string           instruction)
    {
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion)
            .GetInstructionInfo(instruction);
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitectureLibrary::GetInstructionInfo(std::string isaVersionString,
                                                   std::string instruction)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion)
            .GetInstructionInfo(instruction);
    }

    inline GPUArchitecture GPUArchitectureLibrary::GetDevice(std::string isaVersionString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return GPUArchitectureLibrary::GPUArchitectures.at(isaVersion);
    }

    inline std::vector<std::string> GPUArchitectureLibrary::getAllSupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getAllSupportedISAs");

        std::vector<std::string> result;

        for(auto target : GPUArchitectures)
        {
            result.push_back(target.first.ToString());
        }

        return result;
    }

    inline std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectureLibrary::LoadLibrary()
    {
        TIMER(t, "GPUArchitectureLibrary::LoadLibrary");

        char* yaml_file = getenv(ENV_ARCHITECTURE_YAML_FILE.c_str());
        if(yaml_file)
        {
            try
            {
                return GPUArchitecture::readYaml(yaml_file);
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error(
                    "Could not read GPU Architecture Library file specified in env var "
                    + ENV_ARCHITECTURE_YAML_FILE + ": " + std::string(yaml_file));
            }
        }
        else
        {
            try
            {
                return GPUArchitecture::readMsgpack(ARCHITECTURE_MSGPACK_FILE);
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error("Could not read default GPU Architecture Library file: "
                                         + ARCHITECTURE_MSGPACK_FILE);
            }
        }
    }
}
