#pragma once
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <rocRoller/Utilities/Settings.hpp>

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

        std::string archFile = Settings::getInstance()->get(Settings::ArchitectureFile);

        // TODO: Support non-Linux OS
        std::filesystem::path archPath
            = std::filesystem::read_symlink("/proc/self/exe").parent_path();
        archPath /= archFile;

        if(archFile.find(".yaml") != std::string::npos
           || archFile.find(".yml") != std::string::npos)
        {
            return GPUArchitecture::readYaml(archPath.string());
        }
        else
        {
            return GPUArchitecture::readMsgpack(archPath.string());
        }
    }
}