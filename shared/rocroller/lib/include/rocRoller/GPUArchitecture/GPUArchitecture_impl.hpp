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

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/Utilities/Timer.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline constexpr GPUArchitectureTarget const& GPUArchitecture::target() const
    {
        return m_archTarget;
    }

    inline void GPUArchitecture::AddCapability(GPUCapability const& capability, int value)
    {
        m_capabilities[capability] = value;
    }

    inline void GPUArchitecture::AddInstructionInfo(GPUInstructionInfo const& info)
    {
        if(m_instructionInfos.find(info.getInstruction()) != m_instructionInfos.end())
            throw std::runtime_error(
                concatenate("Instruction info already exists for ", info.getInstruction()));
        m_instructionInfos[info.getInstruction()] = info;
    }

    inline bool GPUArchitecture::HasCapability(GPUCapability const& capability) const
    {
        return m_capabilities.find(capability) != m_capabilities.end();
    }

    inline bool GPUArchitecture::HasCapability(std::string const& capabilityString) const
    {
        return m_capabilities.find(GPUCapability(capabilityString)) != m_capabilities.end();
    }

    inline int GPUArchitecture::GetCapability(GPUCapability const& capability) const
    {
        auto iter = m_capabilities.find(capability);
        if(iter == m_capabilities.end())
            throw std::runtime_error(concatenate("Capability ", capability, " not found"));

        return iter->second;
    }

    inline int GPUArchitecture::GetCapability(std::string const& capabilityString) const
    {
        return GetCapability(GPUCapability(capabilityString));
    }

    inline bool GPUArchitecture::HasInstructionInfo(std::string const& instruction) const
    {
        auto iter = m_instructionInfos.find(instruction);
        return iter != m_instructionInfos.end();
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitecture::GetInstructionInfo(std::string const& instruction) const
    {
        auto iter = m_instructionInfos.find(instruction);

        if(iter != m_instructionInfos.end())
        {
            return iter->second;
        }
        else
        {
            return GPUInstructionInfo();
        }
    }

    inline GPUArchitecture::GPUArchitecture() {}

    inline GPUArchitecture::GPUArchitecture(GPUArchitectureTarget const& isaVersion)
        : m_archTarget(isaVersion)
    {
    }

    inline GPUArchitecture::GPUArchitecture(
        GPUArchitectureTarget const&                     isaVersion,
        std::map<GPUCapability, int> const&              capabilities,
        std::map<std::string, GPUInstructionInfo> const& instruction_infos)
        : m_archTarget(isaVersion)
        , m_capabilities(capabilities)
        , m_instructionInfos(instruction_infos)
    {
    }

    inline std::ostream& operator<<(std::ostream& os, GPUCapability const& input)
    {
        os << input.toString();
        return os;
    }

    inline std::istream& operator>>(std::istream& is, GPUCapability& input)
    {
        std::string recvd;
        is >> recvd;
        input = GPUCapability(recvd);
        return is;
    }

    inline std::ostream& operator<<(std::ostream& os, GPUWaitQueueType const& input)
    {
        os << input.toString();
        return os;
    }

    template <std::integral T>
    requires(!std::same_as<bool, T>) bool GPUArchitecture::isSupportedConstantValue(T value) const
    {
        auto range = supportedConstantRange<T>();
        return value >= range.first && value <= range.second;
    }

    template <std::integral T>
    std::pair<T, T> GPUArchitecture::supportedConstantRange() const
    {
        std::pair<T, T> rv;
        rv.second = 64;
        if constexpr(std::signed_integral<T>)
            rv.first = -16;
        else
            rv.first = 0;

        return rv;
    }

    template <std::floating_point T>
    std::unordered_set<T> GPUArchitecture::supportedConstantValues() const
    {
        static_assert(CIsAnyOf<T, float, double, Half, BFloat16, FP8, BF8, FP6, BF6, FP4, E8M0>,
                      "Unsupported floating point type");

        if constexpr(CIsAnyOf<T, BFloat16, FP8, BF8, FP6, BF6, FP4, E8M0>)
        {
            return {};
        }
        // clang-format on
        else
        {
            T one_over_two_pi;

            if constexpr(std::same_as<T, float>)
                one_over_two_pi = 0.15915494f;
            else if constexpr(std::same_as<T, double>)
                one_over_two_pi = 0.15915494309189532;
            else if constexpr(std::same_as<T, Half>)
                one_over_two_pi = 0.1592f;

            return {0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0, one_over_two_pi};
        }
    };

    template <std::floating_point T>
    bool GPUArchitecture::isSupportedConstantValue(T value) const
    {
        static auto supportedValues = supportedConstantValues<T>();

        return supportedValues.find(value) != supportedValues.end();
    }

    inline std::map<GPUCapability, int> const& GPUArchitecture::getAllCapabilities() const
    {
        return m_capabilities;
    }

    inline std::map<std::string, GPUInstructionInfo> const&
        GPUArchitecture::getAllIntructionInfo() const
    {
        return m_instructionInfos;
    }

    inline bool GPUArchitecture::isSupportedScaleBlockSize(int size) const
    {
        return m_capabilities.contains(GPUCapability::HasBlockScaling32) && (size == 32);
    }

    inline bool GPUArchitecture::isSupportedScaleType(DataType type) const
    {
        switch(type)
        {
        case DataType::E8M0:
            return m_capabilities.contains(GPUCapability::HasE8M0Scale);
        default:
            return false;
        }
    }
}
