/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
