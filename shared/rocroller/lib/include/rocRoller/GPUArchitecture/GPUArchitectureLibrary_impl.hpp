// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    inline bool GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget const& isaVersion,
                                                      GPUCapability const&         capability)
    {
        return m_gpuArchitectures.find(isaVersion) != m_gpuArchitectures.end()
               && m_gpuArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline bool GPUArchitectureLibrary::HasCapability(GPUArchitectureTarget const& isaVersion,
                                                      std::string const&           capability)
    {
        return m_gpuArchitectures.find(isaVersion) != m_gpuArchitectures.end()
               && m_gpuArchitectures.at(isaVersion).HasCapability(capability);
    }

    inline int GPUArchitectureLibrary::GetCapability(GPUArchitectureTarget const& isaVersion,
                                                     GPUCapability const&         capability)
    {
        return m_gpuArchitectures.at(isaVersion).GetCapability(capability);
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitectureLibrary::GetInstructionInfo(GPUArchitectureTarget const& isaVersion,
                                                   std::string const&           instruction)
    {
        return m_gpuArchitectures.at(isaVersion).GetInstructionInfo(instruction);
    }

    inline std::vector<GPUArchitectureTarget> GPUArchitectureLibrary::getAllSupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getAllSupportedISAs");

        std::vector<GPUArchitectureTarget> result;

        for(auto const& target : m_gpuArchitectures)
        {
            //cppcheck-suppress useStlAlgorithm
            result.push_back(target.first);
        }

        return result;
    }

    inline std::vector<GPUArchitectureTarget> GPUArchitectureLibrary::getMFMASupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getMFMASupportedISAs");

        std::vector<GPUArchitectureTarget> result;

        for(auto const& target : m_gpuArchitectures)
        {
            if(target.second.HasCapability(GPUCapability::HasMFMA))
                result.push_back(target.first);
        }

        return result;
    }

    inline std::vector<GPUArchitectureTarget> GPUArchitectureLibrary::getWMMASupportedISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getWMMASupportedISAs");

        std::vector<GPUArchitectureTarget> result;

        for(auto const& target : m_gpuArchitectures)
        {
            if(target.second.HasCapability(GPUCapability::HasWMMA))
                result.push_back(target.first);
        }

        return result;
    }

    inline std::vector<GPUArchitectureTarget> GPUArchitectureLibrary::getCDNAISAs()
    {
        TIMER(t, "GPUArchitectureLibrary::getCDNAISAs");
        std::vector<GPUArchitectureTarget> result;

        for(auto const& target : m_gpuArchitectures)
        {
            if((target.first).isCDNAGPU())
                result.push_back(target.first);
        }

        return result;
    }

}
