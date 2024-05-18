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
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    inline bool GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget const& isaVersion,
                                                      GPUCapability const&         capability)
    {
        return m_gpuArchitectures.find(isaVersion) != m_gpuArchitectures.end()
               && m_gpuArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline bool GPUArchitectureLibrary::HasCapability(std::string const&   isaVersionString,
                                                      GPUCapability const& capability)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.find(isaVersion) != m_gpuArchitectures.end()
               && m_gpuArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline bool GPUArchitectureLibrary::HasCapability(std::string const& isaVersionString,
                                                      std::string const& capabilityString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.find(isaVersion) != m_gpuArchitectures.end()
               && m_gpuArchitectures.at(isaVersion).HasCapability(capabilityString);
    }

    inline int GPUArchitectureLibrary::GetCapability(GPUArchitectureTarget const& isaVersion,
                                                     GPUCapability const&         capability)
    {
        return m_gpuArchitectures.at(isaVersion).GetCapability(capability);
    }

    inline int GPUArchitectureLibrary::GetCapability(std::string const&   isaVersionString,
                                                     GPUCapability const& capability)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.at(isaVersion).GetCapability(capability);
    }

    inline int GPUArchitectureLibrary::GetCapability(std::string const& isaVersionString,
                                                     std::string const& capabilityString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.at(isaVersion).GetCapability(capabilityString);
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitectureLibrary::GetInstructionInfo(GPUArchitectureTarget const& isaVersion,
                                                   std::string const&           instruction)
    {
        return m_gpuArchitectures.at(isaVersion).GetInstructionInfo(instruction);
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitectureLibrary::GetInstructionInfo(std::string const& isaVersionString,
                                                   std::string const& instruction)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.at(isaVersion).GetInstructionInfo(instruction);
    }

    inline GPUArchitecture GPUArchitectureLibrary::GetDevice(std::string const& isaVersionString)
    {
        GPUArchitectureTarget isaVersion(isaVersionString);
        return m_gpuArchitectures.at(isaVersion);
    }

    inline std::vector<std::string> GPUArchitectureLibrary::getAllSupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getAllSupportedISAs");

        std::vector<std::string> result;

        for(auto target : m_gpuArchitectures)
        {
            result.push_back(target.first.toString());
        }

        return result;
    }

    inline std::vector<std::string> GPUArchitectureLibrary::getMFMASupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getMFMASupportedISAs");

        std::vector<std::string> result;

        for(auto target : m_gpuArchitectures)
        {
            if(target.second.HasCapability(GPUCapability::HasMFMA))
                result.push_back(target.first.toString());
        }

        return result;
    }

    inline std::map<GPUArchitectureTarget, GPUArchitecture> GPUArchitectureLibrary::LoadLibrary()
    {
        TIMER(t, "GPUArchitectureLibrary::LoadLibrary");

        std::string archFile = Settings::getInstance()->get(Settings::ArchitectureFile);

        if(archFile.find(".yaml") != std::string::npos
           || archFile.find(".yml") != std::string::npos)
        {
            return GPUArchitecture::readYaml(archFile);
        }
        else
        {
            return GPUArchitecture::readMsgpack(archFile);
        }
    }
}
